# -*- coding: utf-8 -*-
"""
土壤采样规划工具 (Soil Sampling Planner)
---------------------------------------------------------
本模块实现了基于无人机能谱数据的自动化土壤采样规划流程。
它结合了地统计学（克里金插值）、计算几何（航线重采样）和空间统计学（分位数分级）原理。

主要特性：
1. 高性能计算：利用 NumPy 向量化运算和内存数据流，极大减少磁盘 I/O 开销。
2. 自适应参数：自动计算变异函数步长，适应不同尺度的地块。
3. 自动处理坐标系转换，并兼容无坐标系的情况
4. 智能布点：通过网格化与距离互斥算法，确保采样点空间分布均匀且具有代表性。
5. 出版级制图：生成高分辨率地图，视觉效果清新美观。
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import rasterio
from rasterio.transform import from_origin, array_bounds, rowcol
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import warnings

# 忽略地理数据处理中常见的非致命警告
warnings.filterwarnings("ignore")
# 设置绘图字体 (适配中文显示)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class PureSoilSampler:
    """
    土壤采样核心处理类
    封装了从数据加载、路径规划、空间插值到最终制图的全流程。
    """

    def __init__(self, 
                 workspace, 
                 points_data, 
                 boundary_data, 
                 value_fields=['total'], 
                 time_field='collection_time', 
                 target_crs="auto"): 
        """
        初始化采样器实例，加载数据并建立基础环境。

        :param workspace: 结果输出的主目录路径
        :param points_data: 包含能谱数据的点矢量 (GeoDataFrame)
        :param boundary_data: 农田边界的多边形矢量 (GeoDataFrame)
        :param value_fields: 需要进行插值分析的字段列表
        :param time_field: 用于按时间顺序生成飞行轨迹的字段名
        :param target_crs: 目标投影坐标系 (默认为 UTM Zone 50N)，用于确保距离计算单位为米
        """
        self.workspace = workspace
        # 确保字段列表格式统一
        if isinstance(value_fields, str): self.value_fields = [value_fields]
        else: self.value_fields = value_fields
            
        self.time_field = time_field

        # ========================================================
        # [逻辑]: 自动投影判断
        # ========================================================
        if target_crs == "auto":
            print("正在基于首个数据点识别最佳投影...")
            # 传入原始点数据，函数内部只取第一个点计算，无需担心性能
            self.target_crs = self._get_best_utm_epsg(points_data)
            print(f"      -> 自动匹配为: {self.target_crs}")
        else:
            self.target_crs = target_crs
            print(f"      -> 使用指定投影: {self.target_crs}")
        # ========================================================
        
        # 定义不同采样类型的比例配置 (高值区:低值区:中值区)
        self.ratio_config = {"hotPoint": 0.4, "coldPoint": 0.4, "normalPoint": 0.2}
        
        # 创建输出目录
        if not os.path.exists(workspace): os.makedirs(workspace)

        print("正在加载数据并统一投影...")
        # 将空间数据统一转换到投影坐标系，这对计算真实的欧氏距离至关重要
        self.gdf_pts = points_data.to_crs(self.target_crs)
        self.gdf_bound = boundary_data.to_crs(self.target_crs)
        
        # 数据完整性校验
        if self.time_field not in self.gdf_pts.columns:
            raise ValueError(f"点数据中缺少时间字段: {self.time_field}")

    def _get_best_utm_epsg(self, gdf):
        """
        [功能]: 根据第一个点的经纬度自动计算 UTM EPSG 代码
        [逻辑]: 自动处理坐标系转换，并兼容无坐标系的情况
        """
        try:
            # 1. 提取数据的第一个点 (GeoDataFrame格式)
            first_pt_gdf = gdf.iloc[[0]]
            
            # 2. 检查原始数据是否有坐标系
            if first_pt_gdf.crs is None:
                print("      [警告] 输入数据丢失坐标系信息(CRS is None)！")
                print("      -> 假设原始坐标为 WGS84 经纬度进行计算...")
                # 假设它是经纬度，手动指定为 WGS84，不进行转换
                first_pt_gdf.set_crs("EPSG:4326", inplace=True)
            else:
                # 如果有坐标系 (比如 CGCS2000)，则执行投影转换
                # 将其转换为 WGS84，以便提取经度(Lon)和纬度(Lat)
                first_pt_gdf = first_pt_gdf.to_crs("EPSG:4326")
            
            # 3. 获取几何点坐标
            pt = first_pt_gdf.geometry.iloc[0]
            lon, lat = pt.x, pt.y
            
            # 4. 计算 UTM 带号 (全球通用公式)
            # floor((经度 + 180) / 6) + 1
            zone_number = int((lon + 180) / 6) + 1
            
            # 5. 判断南北半球
            base = "326" if lat >= 0 else "327"
            
            epsg_code = f"EPSG:{base}{zone_number:02d}"
            
            # 打印调试信息，让你放心
            # print(f"      [Debug] 检测位置: ({lon:.2f}, {lat:.2f}) -> 对应投影: {epsg_code}")
            
            return epsg_code
            
        except Exception as e:
            print(f"      [警告] 自动识别投影失败 ({e})，回退至默认 EPSG:32650")
            return "EPSG:32650"
            
        except Exception as e:
            # 兜底策略：如果数据为空或极度异常，默认返回一个通用值并报警告
            print(f"      [警告] 自动识别投影失败 ({e})，回退至默认 EPSG:32650")
            return "EPSG:32650"

    def _generate_trajectory(self):
        """
        根据时间戳生成无人机的飞行轨迹线。
        用于后续沿着实际飞行路径提取插值结果，而非全图随机提取。
        """
        sorted_pts = self.gdf_pts.sort_values(by=self.time_field)
        # 至少需要两个点才能构成线
        if len(sorted_pts) < 2: return None
        
        line_geom = LineString(sorted_pts.geometry.tolist())
        return gpd.GeoDataFrame({'geometry': [line_geom]}, crs=self.target_crs)

    def _get_target_counts(self, mode, density, min_count, fixed_counts):
        """
        计算各分类需要布设的采样点数量。
        
        :param mode: 'auto' (基于面积计算) 或 'fixed' (使用固定数量)
        :param density: 采样密度 (点/公顷)
        :param min_count: 最小保底点数
        :param fixed_counts: 固定模式下的数量字典
        :return: 各类别的目标数量字典, 总数量
        """
        if mode == "fixed":
            return fixed_counts, sum(fixed_counts.values())
        else:
            # 计算投影面积 (平方米 -> 公顷)
            area_sqm = self.gdf_bound.area.sum()
            area_ha = area_sqm / 10000.0
            
            # 根据密度计算总需求量，并应用保底逻辑
            total_needed = max(min_count, int(math.ceil(area_ha * density)))
            
            # 按预设比例分配各类别数量
            counts = {}
            allocated = 0
            for k, r in self.ratio_config.items():
                c = max(1, int(round(total_needed * r))) # 确保各类至少有1个点
                counts[k] = c
                allocated += c
            
            # 如果分配后仍有剩余名额，优先补给高值区 (hotPoint)
            if allocated < total_needed: 
                counts['hotPoint'] += (total_needed - allocated)
                
            print(f"      [智能参数] 面积:{area_ha:.2f}ha | 计划采样:{sum(counts.values())}")
            return counts, sum(counts.values())

    def _calculate_optimal_nlags(self, x, y):
        """
        利用空间统计学原理自动计算变异函数的分箱数 (nlags)。
        
        原理: 
        将'平均最近邻距离'视为最佳步长 (Lag Size)。
        分箱数 (nlags) = 区域最大跨度 / 平均最近邻距离。
        这能保证变异函数既能捕捉短距离的空间自相关，又不会因步长过小导致计算噪声。
        """
        try:
            coords = np.column_stack((x, y))
            # 使用 KDTree 快速查询最近邻
            tree = cKDTree(coords)
            # k=2 是因为最近的邻居是点本身(距离为0)，我们需要第二个最近的点
            dists, _ = tree.query(coords, k=2)
            avg_nn_dist = np.mean(dists[:, 1])
            
            # 计算点集在空间上的最大对角线距离
            max_span = math.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2)
            
            if avg_nn_dist <= 0: return 6
            
            nlags = int(max_span / avg_nn_dist)
            # 限制 nlags 在合理范围内 (6 ~ 100)，防止极端情况导致计算失败
            nlags = max(6, min(nlags, 100))
            
            print(f"      [自动步长] 平均间距: {avg_nn_dist:.2f}m | 计算分箱(nlags): {nlags}")
            return nlags
        except Exception as e:
            print(f"      [警告] 自动步长计算失败: {e}，将使用默认值。")
            return 6

    def _perform_kriging(self, field_name, resolution):
        """
        执行普通克里金插值 (Ordinary Kriging)。
        使用了向量化后端 (Vectorized Backend) 以大幅提升计算速度。
        
        :param field_name: 目标字段
        :param resolution: 输出栅格分辨率 (米)
        :return: 插值结果矩阵, 仿射变换参数, 坐标系对象
        """
        # 提取坐标与属性值
        x = self.gdf_pts.geometry.x.values
        y = self.gdf_pts.geometry.y.values
        z = self.gdf_pts[field_name].values

        # 1. 自动计算模型参数
        optimal_nlags = self._calculate_optimal_nlags(x, y)

        # 2. 构建插值网格 (设置 pad 防止边缘效应)
        minx, miny, maxx, maxy = self.gdf_bound.total_bounds
        pad = resolution * 10
        grid_x = np.arange(minx - pad, maxx + pad, resolution)
        grid_y = np.arange(miny - pad, maxy + pad, resolution)

        # 3. 初始化克里金模型 (球状模型适用于大多数土壤属性)
        ok = OrdinaryKriging(
            x, y, z, 
            variogram_model='spherical', 
            verbose=False, 
            nlags=optimal_nlags, 
            enable_plotting=False
        )
        
        # 4. 执行网格预测
        # backend='vectorized' 利用 NumPy 矩阵运算替代循环，极大加速计算
        z_vals, _ = ok.execute('grid', grid_x, grid_y, backend='vectorized')
        
        # 翻转 Y 轴数据，因为数学坐标系(Y向上)与图像坐标系(Y向下)相反
        z_vals = np.flipud(z_vals)

        # 5. 构建地理参考转换 (Transform)
        transform = from_origin(grid_x[0], grid_y[-1] + resolution, resolution, resolution)
        
        # 6. 应用掩膜 (将农田边界外的像素置为 NaN)
        shapes = [geom for geom in self.gdf_bound.geometry]
        mask = geometry_mask(shapes, transform=transform, invert=True, out_shape=z_vals.shape)
        z_masked = np.where(mask, z_vals, np.nan)
        
        return z_masked, transform, self.gdf_pts.crs

    def _sample_from_memory(self, gdf_route, z_data, transform):
        """
        内存采样函数。
        直接利用仿射变换原理，将地理坐标映射到内存数组的行列号，提取栅格值。
        避免了传统方法中"写入硬盘 -> 读取硬盘 -> 采样"的 I/O 瓶颈。
        """
        coords = [(p.x, p.y) for p in gdf_route.geometry]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        
        # 批量转换地理坐标到数组索引
        rows, cols = rowcol(transform, xs, ys)
        
        vals = []
        height, width = z_data.shape
        
        for r, c in zip(rows, cols):
            # 边界检查，防止索引越界
            if 0 <= r < height and 0 <= c < width:
                val = z_data[r, c]
                # 检查是否为有效值 (非 NaN)
                if np.isnan(val):
                    vals.append(np.nan)
                else:
                    vals.append(val)
            else:
                vals.append(np.nan)
        
        return vals

    def _plot_result_map(self, region_id, field_name, gdf_samples, gdf_traj, z_masked, transform, src_crs):
        """
        [布局更新]: 绘制可视化图表
        1. 坐标轴: 使用符号化经纬度 (如 49.454°N), 去除汉字标签。
        2. 布局: 图例在地图右侧, Colorbar 在地图正下方(水平放置)。
        """
        print(f"      正在绘制可视化图表...")
        
        # 1. 坐标系转换 (统一转为 WGS84 经纬度用于制图显示)
        bound_wgs = self.gdf_bound.to_crs("EPSG:4326")
        traj_wgs = gdf_traj.to_crs("EPSG:4326") if gdf_traj is not None else None
        samples_wgs = gdf_samples.to_crs("EPSG:4326")
        
        # 2. 栅格重投影 (从投影坐标系重采样为经纬度)
        height, width = z_masked.shape
        dst_crs = 'EPSG:4326'
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, width, height, *array_bounds(height, width, transform)
        )
        destination = np.zeros((dst_height, dst_width), np.float32)
        reproject(
            source=z_masked, destination=destination,
            src_transform=transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan, dst_nodata=np.nan
        )
        
        # 3. 初始化画布 (300 DPI 高清设置)
        fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
        
        # [布局控制]: 调整子图边距
        # right=0.85: 为右侧图例留出约 15% 的宽度
        # bottom=0.15: 为底部水平色条留出约 15% 的高度
        plt.subplots_adjust(right=0.85, bottom=0.15)
        
        # 4. 绘制插值底图
        dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
        plot_extent = [dst_bounds[0], dst_bounds[2], dst_bounds[1], dst_bounds[3]]
        
        # 清新配色
        fresh_colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        fresh_cmap = LinearSegmentedColormap.from_list("fresh_style", fresh_colors)
        im = ax.imshow(destination, cmap=fresh_cmap, extent=plot_extent, alpha=0.85, origin='upper')

        # 5. 绘制矢量元素
        if traj_wgs is not None:
            traj_wgs.plot(ax=ax, color="gray", linewidth=1, alpha=0.5, linestyle="-", label="飞行轨迹", zorder=2)
        bound_wgs.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2, linestyle="--", label="农田边界", zorder=3)
        
        # 6. 绘制采样点
        color_dict = {'hotPoint': 'red', 'coldPoint': 'blue', 'normalPoint': 'yellow'}
        for cls in ['hotPoint', 'normalPoint', 'coldPoint']:
            sub = samples_wgs[samples_wgs['class'] == cls]
            if not sub.empty:
                sub.plot(ax=ax, color=color_dict[cls], edgecolor='white', markersize=80, zorder=10)

        # ========================================================
        # [修改点]: 坐标轴格式化 (符号化经纬度)
        # ========================================================
        
        # 定义经度格式化函数 (保留3位小数，如 120.300°E)
        def format_lon(x, pos):
            return f"{x:.3f}°E" if x >= 0 else f"{abs(x):.3f}°W"

        # 定义纬度格式化函数 (保留3位小数，如 49.454°N)
        def format_lat(y, pos):
            return f"{y:.3f}°N" if y >= 0 else f"{abs(y):.3f}°S"

        # 应用自定义格式器
        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        
        # 移除原有的汉字标签 (经度/纬度)，因为刻度本身已包含方向信息
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_title(f"土壤采样规划图 - {region_id} ({field_name})", fontsize=14, pad=15)
        
        # ========================================================
        # [布局逻辑 A]: 图例放置在右侧
        # ========================================================
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', lw=2, label='农田边界'),
            Line2D([0], [0], color='gray', lw=1, label='飞行轨迹'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='高值点'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='中值点'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='低值点')
        ]
        
        # bbox_to_anchor=(1.02, 1.0): 锚点位于主图右边界外侧，顶部对齐
        ax.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1.02, 1.0), 
                  title="图例", frameon=True, borderpad=1)
        
        # ========================================================
        # [布局逻辑 B]: 色条(Colorbar)放置在正下方
        # ========================================================
        # 获取主图坐标轴的位置 [x0, y0, width, height]
        pos = ax.get_position()
        
        # 计算色条位置: 
        # left/width 与主图对齐
        # bottom 在主图下方偏移 0.09 (避开X轴刻度)
        # height 设为 0.03 (细长条)
        cax_bounds = [pos.x0, pos.y0 - 0.09, pos.width, 0.03]
        
        # 创建子坐标轴并绘制
        cax = fig.add_axes(cax_bounds)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(f'{field_name} 值', labelpad=10)

        # 8. 保存结果
        out_png = os.path.join(self.workspace, f"Map_{region_id}_{field_name}.png")
        # bbox_inches=None 确保手动布局不被自动裁切破坏
        plt.savefig(out_png, dpi=300, bbox_inches=None) 
        plt.close()
        print(f"      -> 图片已保存: {os.path.basename(out_png)}")
    
    def run(self, region_id="Region", mode="auto", density=1.5, min_count=4, fixed_counts=None, resample_dist=5.0, resolution=2.0):
        """
        主控制函数，串联所有处理步骤。
        """
        print(f"\n=== 开始处理区域: {region_id} ===")
        
        # 1. 准备阶段：计算采样数量和生成航线
        target_counts, total_num = self._get_target_counts(mode, density, min_count, fixed_counts)
        gdf_traj = self._generate_trajectory()
        
        # 沿着航线进行重采样，生成候选点集
        if gdf_traj is not None:
            line_geom = gdf_traj.geometry.iloc[0]
            num_points = int(line_geom.length / resample_dist)
            resampled_coords = [line_geom.interpolate(i * resample_dist) for i in range(num_points + 1)]
            gdf_route_pts = gpd.GeoDataFrame(geometry=resampled_coords, crs=self.target_crs)
        else:
            # 如果无法生成航线（点太少），直接使用原始点
            gdf_route_pts = self.gdf_pts.copy()

        # 2. 循环处理每一个监测字段
        for field in self.value_fields:
            print(f"  >>> 处理要素: [{field}] ...")
            try:
                # A. 插值计算 (全内存操作)
                z_masked, transform, src_crs = self._perform_kriging(field, resolution)
                
                # B. 采样提取 (全内存操作)
                vals = self._sample_from_memory(gdf_route_pts, z_masked, transform)
                
                # C. 数据整理与分类
                df_work = gdf_route_pts.copy()
                df_work['val'] = vals
                df_work = df_work[~np.isnan(df_work['val'])] # 剔除无效值
                if len(df_work) == 0: 
                    print("      [警告] 所有候选点均位于插值范围外，跳过该字段。")
                    continue

                # 计算分位点并进行三级分类
                p33 = np.percentile(df_work['val'], 33)
                p66 = np.percentile(df_work['val'], 66)
                def classify(v): return 'coldPoint' if v < p33 else ('normalPoint' if v < p66 else 'hotPoint')
                df_work['class'] = df_work['val'].apply(classify)
                
                # D. 空间互斥抽样算法
                # 目的：在满足数量要求的同时，让采样点在空间上尽可能分散
                final_list = []
                area_m2 = self.gdf_bound.area.sum()
                # 动态计算互斥半径: 约为理论均匀分布间距的 0.6 倍
                min_dist = math.sqrt(area_m2 / (total_num + 1)) * 0.6
                
                for cls, num in target_counts.items():
                    # 随机打乱候选点
                    pool = df_work[df_work['class'] == cls].sample(frac=1).reset_index(drop=True)
                    selected = []
                    for _, row in pool.iterrows():
                        if len(selected) >= num: break
                        pt = row.geometry
                        is_far = True
                        # 检查与当前类已选点及全局已选点的距离
                        for s in selected + final_list:
                            if pt.distance(s['geometry']) < min_dist: is_far = False; break
                        if is_far: selected.append(row.to_dict())
                    
                    # 如果因距离限制导致数量不足，则放宽限制进行补齐
                    if len(selected) < num:
                        needed = num - len(selected)
                        leftover = pool.iloc[len(selected):len(selected)+needed]
                        if not leftover.empty: selected.extend(leftover.to_dict('records'))
                    final_list.extend(selected)
                
                # E. 结果导出
                gdf_final = gpd.GeoDataFrame(final_list, crs=self.target_crs)
                gdf_wgs = gdf_final.to_crs("EPSG:4326")
                
                # 导出 CSV 表格 (包含经纬度坐标)
                csv_path = os.path.join(self.workspace, f"Coords_{region_id}_{field}.csv")
                out_df = pd.DataFrame({
                    'ID': range(1, len(gdf_wgs)+1), 'Class': gdf_wgs['class'],
                    'Value': gdf_wgs['val'].round(2), 'Lon': gdf_wgs.geometry.x.round(6),
                    'Lat': gdf_wgs.geometry.y.round(6)
                })
                out_df.to_csv(csv_path, index=False, encoding='utf_8_sig')
                print(f"      -> 表格已导出: {os.path.basename(csv_path)}")
                
                # F. 存档栅格 (延迟写入，不影响计算流)
                tif_path = os.path.join(self.workspace, f"Kriging_{region_id}_{field}.tif")
                with rasterio.open(
                    tif_path, 'w', driver='GTiff',
                    height=z_masked.shape[0], width=z_masked.shape[1],
                    count=1, dtype='float32', crs=self.target_crs, 
                    transform=transform, nodata=np.nan
                ) as dst:
                    dst.write(z_masked.astype('float32'), 1)
                
                # G. 执行制图
                self._plot_result_map(region_id, field, gdf_final, gdf_traj, z_masked, transform, src_crs)
                
            except Exception as e:
                print(f"      !! 错误: {e}")
                import traceback
                traceback.print_exc()
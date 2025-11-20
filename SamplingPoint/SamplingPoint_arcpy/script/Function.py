import arcpy
import random
import math
import sys
import os
import shutil
import time
import csv
import traceback
import glob
from arcpy.sa import *

# ===============================================================================
# 类: SoilSampler (v13.3 Enhanced Logging)
# ===============================================================================
class SoilSampler:
    """
    SoilSampler v13.3
    
    [功能描述]:
    自动化土壤采样规划工具。基于无人机能谱数据，结合地统计学原理，生成空间分布均匀且具有代表性的采样方案。
    
    [核心特性]:
    1. [多要素并行]: 可同时处理 Total, K40, Bi214 等多个指标，互不干扰。
    2. [自动步长]: 集成 '平均最近邻' 算法，自动计算克里金插值的最佳步长(Lag Size)。
    3. [栅格直出]: 插值结果直接保存为文件夹中的 .tif 文件，彻底解决数据库文件名截断问题。
    4. [智能抽样]: 使用 '网格分箱 + 距离约束' 算法。如果网格法未收敛，自动切换至 '距离约束随机' 模式，并打印详细日志。
    """

    def __init__(self, 
                 workspace,
                 batch_config,
                 output_utm_sr,
                 
                 # --- 数据源参数 ---
                 value_fields=["total"],        # 要处理的能谱字段列表
                 time_field="collection_time",  # 用于生成航线的时间字段
                 
                 # --- 智能采样参数 ---
                 mode="auto",            # "auto"(按面积算) 或 "fixed"(固定数量)
                 sampling_density=1.5,   # 密度: 每公顷采样的点数
                 min_sample_count=4,     # 保底: 无论地块多小，最少采样的点数
                 
                 # 比例配置
                 ratio_config={          
                     "hotPoint": 0.4,    # 高值区
                     "coldPoint": 0.4,   # 低值区
                     "normalPoint": 0.2  # 中值区
                 },
                 
                 # --- 固定模式参数 ---
                 fixed_counts={"hotPoint": 5, "coldPoint": 5, "normalPoint": 3},
                 fixed_grid_size=100,    
                 
                 # --- 算法参数 ---
                 kriging_model_str="SPHERICAL", 
                 kriging_range="AUTO",          
                 line_point_distance="5 Meters",
                 classification_method="quantile", 
                 symbology_template_lyrx=None,  
                 overwrite=True):               
        
        self.workspace = workspace
        self.overwrite = overwrite
        
        # 定义输出文件夹
        self.grid_folder = os.path.join(os.path.dirname(workspace), "Grid")
        self.scratch_folder = os.path.join(os.path.dirname(workspace), "_scratch_trash")
        
        # 初始化环境
        self._setup_environment()
        
        if isinstance(value_fields, str): self.value_fields = [value_fields]
        else: self.value_fields = value_fields
            
        self.batch_config = batch_config
        self.output_utm_sr = output_utm_sr
        self.mode = mode
        self.sampling_density = sampling_density
        self.min_sample_count = min_sample_count
        self.ratio_config = ratio_config
        self.fixed_counts = fixed_counts
        self.fixed_grid_size = fixed_grid_size
        self.time_field = time_field
        self.line_point_distance = line_point_distance
        self.classification_method = classification_method
        self.symbology_template_lyrx = symbology_template_lyrx
        self.kriging_model_str = kriging_model_str
        self.kriging_range_param = kriging_range 
        self.utm_sr = self._resolve_output_sr(output_utm_sr)

    def _setup_environment(self):
        """初始化 ArcPy 环境与文件夹"""
        try:
            if not os.path.exists(self.grid_folder): os.makedirs(self.grid_folder)
            if not os.path.exists(self.scratch_folder): os.makedirs(self.scratch_folder)
            
            arcpy.env.workspace = self.workspace
            # 强制临时空间指向物理文件夹
            arcpy.env.scratchWorkspace = self.scratch_folder
            arcpy.env.overwriteOutput = self.overwrite
            
            print(f"工作空间(GDB): {self.workspace}")
            print(f"栅格输出(Grid): {self.grid_folder}")
        except Exception as e:
            raise Exception(f"设置工作空间失败: {e}")

    def _resolve_output_sr(self, sr_input):
        if not sr_input: return None
        try: return arcpy.SpatialReference(sr_input)
        except: return None

    def _cleanup_junk_files(self):
        """清理残留文件"""
        try:
            arcpy.env.workspace = self.workspace
            junk = arcpy.ListRasters("Kriging_*")
            if junk:
                for r in junk:
                    try: arcpy.management.Delete(r)
                    except: pass
            
            if os.path.exists(self.grid_folder):
                for f in os.listdir(self.grid_folder):
                    if f.startswith("t_") and f.endswith(".tif"):
                        try: os.remove(os.path.join(self.grid_folder, f))
                        except: pass
                        
            if os.path.exists(self.scratch_folder):
                try: shutil.rmtree(self.scratch_folder)
                except: pass
        except: pass

    def _calculate_optimal_lag_size(self, point_fc):
        """自动计算最佳步长"""
        try:
            result = arcpy.stats.AverageNearestNeighbor(point_fc, "EUCLIDEAN_DISTANCE")
            avg_distance = float(result.getOutput(4))
            return round(avg_distance, 6)
        except Exception as e:
            print(f"    !! 自动计算步长失败: {e}，使用默认值 300。")
            return 300.0

    def _calculate_smart_params(self, farmland_fc, field_name):
        """
        [算法] 智能计算采样数量
        """
        area_sq_meters = 0
        # 使用测地线面积计算真实的地球面积
        with arcpy.da.SearchCursor(farmland_fc, ["SHAPE@"]) as cursor:
            for row in cursor:
                geom = row[0]
                area_sq_meters += geom.getArea("GEODESIC", "SQUAREMETERS")
        
        area_ha = area_sq_meters / 10000.0
        
        total_needed = int(math.ceil(area_ha * self.sampling_density))
        if total_needed < self.min_sample_count:
            total_needed = self.min_sample_count
            
        current_counts = {}
        allocated = 0
        for cls, ratio in self.ratio_config.items():
            c = int(round(total_needed * ratio))
            if c < 1: c = 1 
            current_counts[cls] = c
            allocated += c
            
        if allocated < total_needed:
             k = list(current_counts.keys())[0]
             current_counts[k] += (total_needed - allocated)
            
        if total_needed > 0:
            suggested_grid = math.sqrt(area_sq_meters / total_needed) * 0.8
        else:
            suggested_grid = 100
            
        # [日志] 打印智能计算结果
        print(f"      [智能参数] 农田:{area_ha:.2f}ha | 计划:{sum(current_counts.values())}点 | 目标间距(网格):{suggested_grid:.1f}m")
        return current_counts, suggested_grid

    def _get_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _is_valid_distance(self, new_point, existing_points, min_dist):
        """[算法] 检查新点与已选点集的距离是否合格"""
        for ep in existing_points:
            dist = self._get_distance((new_point["x"], new_point["y"]), (ep["x"], ep["y"]))
            if dist < min_dist: return False
        return True

    def _sample_via_grid_binning(self, input_fc, counts_config, start_grid_size):
        """
        [核心算法] 网格分箱抽样 + 距离约束保底
        """
        data_pool = []
        with arcpy.da.SearchCursor(input_fc, ["OID@", "SHAPE@XY", "class"]) as cursor:
            for row in cursor:
                if row[2] in counts_config: 
                    data_pool.append({"oid": row[0], "x": row[1][0], "y": row[1][1], "cls": row[2]})
        if not data_pool: return []

        total_needed = sum(counts_config.values())
        
        # 硬性最小距离
        HARD_MIN_DISTANCE = start_grid_size * 0.5
        final_oids = []
        
        # 1. 自适应网格尝试
        current_grid = start_grid_size
        for attempt in range(15):
            grid = {}
            for pt in data_pool:
                k = (int(pt["y"]/current_grid), int(pt["x"]/current_grid))
                if k not in grid: grid[k] = {cls: [] for cls in counts_config}
                if pt["cls"] in grid[k]: grid[k][pt["cls"]].append(pt) 

            if len(grid) < total_needed:
                current_grid *= 0.85; continue

            round_points = []
            round_oids = []
            available_keys = list(grid.keys())
            priority = ["hotPoint", "coldPoint", "normalPoint"]
            success = True
            
            for cls in priority:
                if cls not in counts_config: continue
                needed = counts_config[cls]
                candidates = [k for k in available_keys if len(grid[k][cls]) > 0]
                
                if len(candidates) < needed:
                    success = False; break
                
                picked_keys = random.sample(candidates, needed)
                for k in picked_keys:
                    chosen_pt = random.choice(grid[k][cls])
                    if self._is_valid_distance(chosen_pt, round_points, HARD_MIN_DISTANCE):
                        round_points.append(chosen_pt)
                        round_oids.append(chosen_pt["oid"])
                        available_keys.remove(k)
                    else:
                        success = False; break
                if not success: break
            
            if success and len(round_oids) == total_needed:
                final_oids = round_oids
                # print(f"      -> 网格法成功! 最小间距 > {HARD_MIN_DISTANCE:.1f}m")
                break
            else:
                current_grid *= 0.85

        # 2. 保底机制 (如果网格法失败)
        if not final_oids:
            # [日志] 打印切换提示
            print(f"      -> 网格法未收敛，切换至[距离约束随机]模式...")
            print(f"      -> 正在寻找满足 {HARD_MIN_DISTANCE:.1f}米 间距的组合...")
            
            for retry in range(50): 
                temp_pts = []; temp_ids = []; failed = False
                for cls in ["hotPoint", "coldPoint", "normalPoint"]:
                    if cls not in counts_config: continue
                    needed = counts_config[cls]
                    pool = [p for p in data_pool if p["cls"] == cls]
                    random.shuffle(pool)
                    cnt = 0
                    for p in pool:
                        if self._is_valid_distance(p, temp_pts, HARD_MIN_DISTANCE):
                            temp_pts.append(p); temp_ids.append(p["oid"]); cnt += 1
                        if cnt == needed: break
                    if cnt < needed: failed = True; break
                if not failed: 
                    final_oids = temp_ids
                    # [日志] 打印成功尝试次数
                    print(f"      -> 成功找到方案! (尝试第 {retry+1} 次)")
                    break
            
            if not final_oids:
                print(f"      !! 警告: 无法满足 {HARD_MIN_DISTANCE:.1f}m 间距。执行纯随机。")
                for cls, num in counts_config.items():
                     oids = [p["oid"] for p in data_pool if p["cls"] == cls]
                     final_oids.extend(random.sample(oids, min(len(oids), num)))

        return final_oids

    def run_batch(self):
        """[主程序] 批量处理入口"""
        print(f"开始批量处理 (v13.3 日志增强版)...")
        self._cleanup_junk_files() # 预清理
        if not os.path.exists(self.scratch_folder): os.makedirs(self.scratch_folder)

        for region in self.batch_config:
            region_id = region["id"]
            print(f"\n=== 正在处理区域: {region_id} ===")
            
            # 1. 基础几何
            uav_line_base = f"Route_{region_id}"
            line_points_base = f"Points_{region_id}"
            try:
                if arcpy.Exists(uav_line_base): arcpy.management.Delete(uav_line_base)
                arcpy.management.PointsToLine(region["points"], uav_line_base, "", self.time_field)
                if arcpy.Exists(line_points_base): arcpy.management.Delete(line_points_base)
                arcpy.management.GeneratePointsAlongLines(uav_line_base, line_points_base, "DISTANCE", self.line_point_distance)
            except Exception as e:
                print(f"!! 基础几何生成失败: {e}"); continue

            # 2. 自动步长
            current_lag_size = 300.0
            if self.kriging_range_param == "AUTO":
                print(f"  [自动分析] 计算点间距...")
                current_lag_size = self._calculate_optimal_lag_size(region["points"])
                print(f"  [自动分析] 最佳步长: {current_lag_size:.4f} 米")
            else:
                current_lag_size = float(self.kriging_range_param)

            current_kriging_model = KrigingModelOrdinary(self.kriging_model_str, current_lag_size)

            # --- 遍历要素 ---
            for field_name in self.value_fields:
                print(f"\n  >>> 正在处理要素: [{field_name}] ...")
                
                kriging_tif_name = f"Kriging_{region_id}_{field_name}.tif"
                kriging_tif_path = os.path.join(self.grid_folder, kriging_tif_name)
                
                working_points = f"WorkPt_{region_id}_{field_name}"
                projected_points = f"ProjPt_{region_id}_{field_name}"
                final_sample_fc = f"Sample_{region_id}_{field_name}"

                try:
                    # A. 插值
                    arcpy.env.extent = region["farmland"]
                    arcpy.env.mask = region["farmland"]
                    
                    try: 
                        if os.path.exists(kriging_tif_path):
                            try: arcpy.management.Delete(kriging_tif_path)
                            except: os.remove(kriging_tif_path)
                        
                        krig_obj = Kriging(region["points"], field_name, current_kriging_model)
                        arcpy.management.CopyRaster(krig_obj, kriging_tif_path, pixel_type="32_BIT_FLOAT", format="TIFF")
                        
                        # [日志] 更新插值成功提示
                        print(f"      -> 插值成功: {kriging_tif_name}")
                        
                        try: arcpy.management.CalculateStatistics(kriging_tif_path)
                        except: pass
                    except Exception as e:
                        print(f"      !! 插值失败: {e}"); raise
                    finally: 
                        arcpy.env.mask = ""; arcpy.env.extent = ""

                    # B. 提取
                    if arcpy.Exists(working_points): arcpy.management.Delete(working_points)
                    arcpy.management.CopyFeatures(line_points_base, working_points)
                    ExtractMultiValuesToPoints(working_points, [[kriging_tif_path, "krig_val"]], "NONE")
                    
                    vals = [row[0] for row in arcpy.da.SearchCursor(working_points, ["krig_val"]) if row[0] is not None]
                    if not vals: continue
                    vals.sort()
                    p33, p66 = vals[int(len(vals)*0.33)], vals[int(len(vals)*0.66)]
                    
                    arcpy.management.AddField(working_points, "class", "TEXT")
                    with arcpy.da.UpdateCursor(working_points, ["krig_val", "class"]) as cursor:
                        for row in cursor:
                            if row[0] is None: row[1] = "NoData"
                            elif row[0] < p33: row[1] = "coldPoint"
                            elif row[0] < p66: row[1] = "normalPoint"
                            else: row[1] = "hotPoint"
                            cursor.updateRow(row)
                    
                    # C. 抽样
                    if arcpy.Exists(projected_points): arcpy.management.Delete(projected_points)
                    arcpy.management.Project(working_points, projected_points, self.utm_sr)
                    
                    if self.mode == "auto": t_counts, start_grid = self._calculate_smart_params(region["farmland"], field_name)
                    else: t_counts, start_grid = self.fixed_counts, self.fixed_grid_size
                    
                    sel_oids = self._sample_via_grid_binning(projected_points, t_counts, start_grid)
                    if not sel_oids: continue

                    # D. 导出
                    print(f"      4. 导出采样点...")
                    sql = f"OBJECTID IN ({','.join(map(str, sel_oids))})"
                    if arcpy.Exists("tmp_exp"): arcpy.management.Delete("tmp_exp")
                    arcpy.management.MakeFeatureLayer(projected_points, "tmp_exp", sql)
                    if arcpy.Exists(final_sample_fc): arcpy.management.Delete(final_sample_fc)
                    arcpy.management.CopyFeatures("tmp_exp", final_sample_fc)
                    arcpy.management.Delete("tmp_exp")

                    # E. 样式
                    if self.symbology_template_lyrx and arcpy.Exists(self.symbology_template_lyrx):
                        time.sleep(0.5); arcpy.ClearWorkspaceCache_management()
                        out_lyrx = os.path.join(os.path.dirname(self.workspace), f"{final_sample_fc}.lyrx")
                        tpl_copy = os.path.join(os.path.dirname(self.workspace), f"Tpl_{region_id}_{field_name}.lyrx")
                        shutil.copy2(self.symbology_template_lyrx, tpl_copy)
                        tmp_lyr = f"{final_sample_fc}_Layer"
                        if arcpy.Exists(tmp_lyr): arcpy.management.Delete(tmp_lyr)
                        full_p = os.path.join(self.workspace, final_sample_fc)
                        arcpy.management.MakeFeatureLayer(full_p, tmp_lyr)
                        arcpy.management.ApplySymbologyFromLayer(tmp_lyr, tpl_copy)
                        arcpy.management.SaveToLayerFile(tmp_lyr, out_lyrx, "ABSOLUTE")
                        arcpy.management.Delete(tmp_lyr)
                        if os.path.exists(tpl_copy): os.remove(tpl_copy)
                        print(f"      -> 样式已应用: {os.path.basename(out_lyrx)}")

                    # F. CSV
                    try:
                        csv_name = f"Coords_{region_id}_{field_name}.csv"
                        csv_path = os.path.join(os.path.dirname(self.workspace), csv_name)
                        wgs84 = arcpy.SpatialReference(4326)
                        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow(["ID", "Class", "Lon", "Lat", f"Value_{field_name}"]) 
                            fp = os.path.join(self.workspace, final_sample_fc)
                            with arcpy.da.SearchCursor(fp, ["OID@", "SHAPE@", "class", "krig_val"]) as cur:
                                for r in cur:
                                    pt = r[1].projectAs(wgs84).firstPoint
                                    v = f"{r[3]:.2f}" if r[3] else "0"
                                    writer.writerow([r[0], r[2], f"{pt.X:.6f}", f"{pt.Y:.6f}", v])
                        print(f"      -> 表格已导出: {csv_name}")
                    except Exception as e: print(f"      (CSV错误: {e})")

                except Exception as e:
                    print(f"    !! 处理失败: {e}"); arcpy.env.mask = ""; continue
        
        self._cleanup_junk_files()
        print("\n!! 采样任务全部完成 !!")










# ===============================================================================
# SoilMapper (可视化制图逻辑)
# 负责：打开 APRX，加载数据，调整范围，导出图片
# ===============================================================================
class SoilMapper:
    """
    SoilMapper v1.0
    
    功能:
    1. 读取 SoilSampler 生成的 GDB 数据。
    2. 将图层添加到 ArcGIS Pro 工程的地图中。
    3. 导出布局为图片 (JPG/PNG)。
    """

    def __init__(self, aprx_path, map_name="Map", layout_name="Layout"):
        self.aprx_path = aprx_path
        self.map_name = map_name
        self.layout_name = layout_name
        self.aprx = None
        
        if not os.path.exists(self.aprx_path):
            raise Exception(f"APRX 工程文件未找到: {self.aprx_path}")
        
        try:
            self.aprx = arcpy.mp.ArcGISProject(self.aprx_path)
            print(f"制图工具初始化成功: {os.path.basename(self.aprx_path)}")
        except Exception as e:
            raise Exception(f"无法加载 APRX: {e}")

    def generate_map(self, gdb_path, region_id, field_name, output_folder, farmland_fc_path):
        """
        为指定的区域和要素生成地图图片
        """
        print(f"  [制图] 正在绘制: {region_id} - {field_name}")
        
        # 1. 构造文件名
        kriging_name = f"kriging_{region_id}_{field_name}"
        route_name = f"uav_route_{region_id}_Base" # 公共几何
        sample_name = f"sample_points_{region_id}_{field_name}"
        
        # 2. 获取 Map 对象
        m = self.aprx.listMaps(self.map_name)[0]
        
        # 3. 清理旧图层 (保留底图)
        # 策略：删除名字里包含 kriging_, uav_route_, sample_points_ 的图层
        for lyr in m.listLayers():
            if any(x in lyr.name for x in ["kriging_", "uav_route_", "sample_points_", "Farmland"]):
                m.removeLayer(lyr)
        
        # 4. 添加图层 (注意顺序：先底后顶，addLayer默认加在顶层，所以我们倒序加，或者用 addDataFromPath)
        # 我们希望顺序 (从下到上): 插值图 -> 农田边界 -> 飞行路径 -> 采样点
        
        # A. 添加插值图 (最底层)
        try:
            krig_lyr = m.addDataFromPath(os.path.join(gdb_path, kriging_name))
            # 简单的拉伸渲染
            if krig_lyr.isRasterLayer:
                sym = krig_lyr.symbology
                if hasattr(sym, 'stretchType'):
                    sym.stretchType = "PercentClip"
                    # 尝试设置色带 (如果报错则忽略)
                    try: sym.colorRamp = self.aprx.listColorRamps("Precipitation")[0]
                    except: pass
                    krig_lyr.symbology = sym
        except: print(f"    警告: 无法添加插值图 {kriging_name}")

        # B. 添加农田边界
        try:
            farm_lyr = m.addDataFromPath(farmland_fc_path)
            # 设置为空心，黑色轮廓
            if farm_lyr.isFeatureLayer:
                sym = farm_lyr.symbology
                sym.renderer.symbol.color = {'RGB': [0, 0, 0, 0]} # 透明填充
                sym.renderer.symbol.outlineColor = {'RGB': [0, 0, 0, 100]} # 黑色轮廓
                sym.renderer.symbol.outlineWidth = 1.5
                farm_lyr.symbology = sym
        except: pass

        # C. 添加飞行路径
        try:
            route_lyr = m.addDataFromPath(os.path.join(gdb_path, route_name))
            if route_lyr.isFeatureLayer:
                sym = route_lyr.symbology
                sym.renderer.symbol.color = {'RGB': [100, 100, 100, 60]} # 灰色
                sym.renderer.symbol.width = 0.5
                route_lyr.symbology = sym
        except: pass

        # D. 添加采样点 (最顶层)
        try:
            # 尝试查找有没有现成的 .lyrx 文件 (由 SoilSampler 生成的)
            lyrx_path = os.path.join(os.path.dirname(gdb_path), f"{sample_name}.lyrx")
            
            if os.path.exists(lyrx_path):
                # 如果有样式文件，直接添加样式文件
                m.addDataFromPath(lyrx_path)
            else:
                # 如果没有，添加数据并设为红色大点
                pt_lyr = m.addDataFromPath(os.path.join(gdb_path, sample_name))
                if pt_lyr.isFeatureLayer:
                    sym = pt_lyr.symbology
                    sym.renderer.symbol.color = {'RGB': [255, 0, 0, 100]}
                    sym.renderer.symbol.size = 10
                    pt_lyr.symbology = sym
        except: print(f"    警告: 无法添加采样点 {sample_name}")

        # 5. 调整布局视图范围
        l = self.aprx.listLayouts(self.layout_name)[0]
        mf = l.listElements("MAPFRAME_ELEMENT")[0]
        mf.map = m # 确保关联正确
        
        # 缩放到农田边界 (farm_lyr)
        # 需要重新获取一下图层对象，因为刚刚添加进去
        layers = m.listLayers()
        target_layer = None
        for lyr in layers:
            # 找刚才添加的边界层
            if lyr.supports("DataSource") and lyr.dataSource == farmland_fc_path:
                target_layer = lyr
                break
        
        if target_layer:
            mf.camera.setExtent(mf.getLayerExtent(target_layer, False, True))
            mf.camera.scale *= 1.2 # 稍微缩小一点，留出边距
        
        # 6. 导出
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        out_jpg = os.path.join(output_folder, f"Map_{region_id}_{field_name}.jpg")
        l.exportToJPEG(out_jpg, resolution=150)
        print(f"    -> 图片已导出: {out_jpg}")

    def save_project(self):
        self.aprx.save()












































"""
# SoilSampler v2.0
# 一个用于UAV能谱数据批量处理、插值、分类和采样点生成的工具类。
# 工作流程:
# 1. 克里金插值 (Kriging)
# 2. 生成UAV航线 (PointsToLine)
# 3. 沿线布点 (GeneratePointsAlongLines)
# 4. 提取插值 (ExtractMultiValuesToPoints)
# 5. 分类 (Classify - 可选方法)
# 6. 投影 (Project)
# 7. 带最小距离约束和回退策略的采样 (Sample)
# 8. 合并导出并(可选)应用符号系统 (Export & Symbolize)
"""
# import arcpy
# import random
# import sys
# import os
# from arcpy.sa import *

# # 这是一个可选依赖，仅在 'std_dev' 分类法中需要
# try:
#     import numpy as np
# except ImportError:
#     print("警告: Numpy 库未找到。")
#     print("      'std_dev' (标准差) 分类方法将不可用。")
#     np = None

# class SoilSampler:
#     """
#     SoilSampler v2.0
#     一个用于UAV能谱数据批量处理、插值、分类和采样点生成的工具类。
    
#     工作流程:
#     1. 克里金插值 (Kriging)
#     2. 生成UAV航线 (PointsToLine)
#     3. 沿线布点 (GeneratePointsAlongLines)
#     4. 提取插值 (ExtractMultiValuesToPoints)
#     5. 分类 (Classify - 可选方法)
#     6. 投影 (Project)
#     7. 带最小距离约束和回退策略的采样 (Sample)
#     8. 合并导出并(可选)应用符号系统 (Export & Symbolize)
#     """

#     def __init__(self, 
#                  workspace,
#                  batch_config,
#                  output_utm_sr,
                 
#                  # --- 核心参数 ---
#                  time_field="collection_time", 
#                  value_field="total", 
                 
#                  # --- 改进 1: 参数化模型 ---
#                  kriging_model_str="SPHERICAL",
#                  kriging_range=300,
#                  line_point_distance="5 Meters",
                 
#                  # --- 改进 2: 灵活分类 ---
#                  classification_method="quantile",
#                  classification_std_dev_factor=0.5,
#                  classification_thresholds=None,
                 
#                  # --- 改进 3: 采样策略 ---
#                  min_distance=100,
#                  sampling_fallback_distances=None,
#                  sample_config={
#                      "coldPoint": 5,
#                      "normalPoint": 3,
#                      "hotPoint": 5,
#                  },
                 
#                  # --- 改进 4: 符号系统 ---
#                  symbology_template_lyrx=None,
                 
#                  # --- 其他 ---
#                  overwrite=True):
#         """
#         初始化土壤采样器 v2.0。
        
#         参数:
#         - workspace (str): ArcPy 工作空间 GDB 路径。
#         - batch_config (list): 区域配置字典的列表。
#         - output_utm_sr (str/int): 必须提供的输出投影坐标系 (例如 "WGS 1984 UTM Zone 50N")。
        
#         - kriging_model_str (str): 克里金模型 ("SPHERICAL", "EXPONENTIAL" 等)。
#         - kriging_range (float): 克里金变程。
#         - line_point_distance (str): 沿线布点距离 (例如 "5 Meters")。
        
#         - classification_method (str): 分类方法: "quantile" (分位数), "std_dev" (标准差), "manual" (手动)。
#         - classification_std_dev_factor (float): 用于 'std_dev' 的标准差倍数 (例如 0.5 表示 Mean +/- 0.5*STD)。
#         - classification_thresholds (dict): 用于 'manual' 的阈值 (例如 {"p33": 1000, "p66": 3000})。

#         - min_distance (float): 采样点的首选最小距离。
#         - sampling_fallback_distances (list): 回退距离列表。如果为 None, 将自动设为 [min_distance, min_distance*0.5, 0]。
#         - sample_config (dict): 每个类别的采样数量。

#         - symbology_template_lyrx (str): 指向 .lyrx 样式文件的路径 (唯一值渲染 'class' 字段)。
#         - overwrite (bool): 是否覆盖现有输出。
#         """
#         self.workspace = workspace
#         self.batch_config = batch_config
#         self.output_utm_sr = output_utm_sr # 强制要求
#         self.time_field = time_field
#         self.value_field = value_field
        
#         # 改进 1
#         self.kriging_model_str = kriging_model_str
#         self.kriging_range = kriging_range
#         self.kriging_model = KrigingModelOrdinary(kriging_model_str, kriging_range)
#         self.line_point_distance = line_point_distance
        
#         # 改进 2
#         self.classification_method = classification_method
#         self.classification_std_dev_factor = classification_std_dev_factor
#         self.classification_thresholds = classification_thresholds
        
#         # 改进 3
#         self.min_distance = min_distance
#         self.sample_config = sample_config
#         if sampling_fallback_distances:
#             self.sampling_fallback_distances = sampling_fallback_distances
#         else:
#             self.sampling_fallback_distances = [self.min_distance, self.min_distance * 0.5, 0] # 默认回退策略
        
#         # 改进 4
#         self.symbology_template_lyrx = symbology_template_lyrx
        
#         self.overwrite = overwrite
        
#         # 1. 设置 ArcPy 环境
#         self._setup_environment()
        
#         # 2. 解析坐标系
#         self.utm_sr = self._resolve_output_sr(self.output_utm_sr)
#         if not self.utm_sr:
#             raise Exception("无法解析输出投影坐标系 (UTM SR)。请在初始化时显式提供 'output_utm_sr' 参数。")
    
#     def _setup_environment(self):
#         """设置 ArcPy 环境变量。"""
#         try:
#             arcpy.env.workspace = self.workspace
#             arcpy.env.overwriteOutput = self.overwrite
#             print(f"工作空间设置为: {self.workspace}")
#         except Exception as e:
#             raise Exception(f"设置工作空间失败: {e}")

#     def _resolve_output_sr(self, sr_input):
#         """解析并验证用于距离计算的投影坐标系 (PCS)。"""
#         if not sr_input:
#             print("错误: 必须在初始化时提供 'output_utm_sr'。")
#             return None
#         try:
#             sr = arcpy.SpatialReference(sr_input)
#             print(f"使用用户指定的坐标系: {sr.name}")
#             if sr.type != "Projected":
#                 print(f"警告: 您指定的坐标系 '{sr.name}' 不是投影坐标系 (PCS)。距离计算可能不准确。")
#             return sr
#         except Exception as e:
#             print(f"错误: 无法从输入 '{sr_input}' 创建坐标系: {e}。")
#             return None
    
#     def _sample_with_min_distance(self, input_fc, where_clause, num_to_sample):
#         """
#         (改进 3: 带回退策略的采样)
#         执行带最小距离约束的抽样。
#         返回：选中的 OID 列表。
#         """
#         candidates = []
#         try:
#             with arcpy.da.SearchCursor(input_fc, ["OID@", "SHAPE@"], where_clause) as cursor:
#                 for row in cursor:
#                     candidates.append(row) # (oid, geometry)
#         except Exception as e:
#             print(f"    错误: 搜索 {input_fc} (Where: {where_clause}) 时出错: {e}")
#             return []

#         if not candidates:
#             print(f"    警告：类别 '{where_clause}' 没有找到任何点。")
#             return []

#         if len(candidates) < num_to_sample:
#             print(f"    警告：类别 '{where_clause}' 点数不足 ({len(candidates)})，无法抽取 {num_to_sample} 个。")
#             num_to_sample = len(candidates) # 调整目标为所有点
        
#         # 随机打乱
#         random.shuffle(candidates)

#         final_oids = [] # 最终 OID 列表
        
#         # --- 改进 3：回退循环 ---
#         for dist in self.sampling_fallback_distances:
#             print(f"    ...正在尝试 {dist}m 最小距离...")
            
#             final_oids = []
#             final_geometries = []
            
#             # 迭代筛选
#             for oid, geom in candidates:
#                 is_valid = True
#                 for existing_geom in final_geometries:
#                     distance = geom.distanceTo(existing_geom)
#                     if distance < dist:
#                         is_valid = False
#                         break
                
#                 if is_valid:
#                     final_oids.append(oid)
#                     final_geometries.append(geom)

#                 if len(final_oids) == num_to_sample:
#                     break # 抽够了

#             # 检查是否在此距离下成功
#             if len(final_oids) == num_to_sample:
#                 print(f"    成功：在 {dist}m 距离下抽出 {num_to_sample} 个点。")
#                 return final_oids # 成功，返回
            
#             # 如果没成功，循环将继续，尝试更小的距离
#             print(f"    未能抽出 {num_to_sample} 个点 (只抽出 {len(final_oids)} 个)。正在尝试下一个回退距离...")

#         # --- 循环结束 ---
#         # 如果所有距离 (包括 0) 都尝试完，返回最后一次尝试的结果
#         print(f"    警告：已尝试所有回退距离。最终返回 {len(final_oids)} 个点 (最后一次尝试使用 {dist}m 距离)。")
#         return final_oids

#     def run_batch(self):
#         """
#         执行所有区域的批量处理。
#         """
#         print(f"开始批量处理 {len(self.batch_config)} 个区域...")
#         print(f"采样配置: {self.sample_config}")
#         print(f"分类方法: {self.classification_method}")
#         print(f"采样距离策略: {self.sampling_fallback_distances}")

#         for region in self.batch_config:
            
#             region_id = region["id"]
#             points = region["points"]
#             farmland = region["farmland"]
            
#             print("\n" + "="*50)
#             print(f"正在处理区域: {region_id}")
#             print(f"  输入点: {points}")
#             print(f"  农田范围: {farmland}")
#             print("="*50)

#             # --- 动态定义本区域的输出名称 ---
#             kriging_raster = f"kriging_total_{region_id}"
#             uav_line = f"uav_route_{region_id}"
#             line_points = f"route_points_{region_id}"
#             projected_line_points = f"route_points_UTM_{region_id}"
#             merged_output_fc = f"sample_points_{region_id}"
            
#             try:
#                 # --- 步骤 1: 克里金插值 (改进 1) ---
#                 print(f"  步骤 1: 克里金插值 (模型: {self.kriging_model_str}, 变程: {self.kriging_range})...")
#                 arcpy.env.extent = farmland
#                 arcpy.env.mask = farmland
#                 try:
#                     kriging_result = Kriging(points, self.value_field, self.kriging_model)
#                     kriging_result.save(kriging_raster)
#                     print(f"    输出: {kriging_raster}")
#                 except Exception as e:
#                     print(f"    !! 步骤 1 失败 (克里金插值): {e}")
#                     raise 
#                 finally:
#                     arcpy.env.mask = ""
#                     arcpy.env.extent = ""

#                 # --- 步骤 2: 点转线 ---
#                 print(f"  步骤 2: 点转线...")
#                 arcpy.management.PointsToLine(points, uav_line, "", self.time_field)

#                 # --- 步骤 3: 沿线生成点 (改进 1) ---
#                 print(f"  步骤 3: 沿线生成点 (距离: {self.line_point_distance})...")
#                 arcpy.management.GeneratePointsAlongLines(
#                     uav_line, line_points,
#                     Point_Placement="DISTANCE", Distance=self.line_point_distance
#                 )

#                 # --- 步骤 4: 值提取至点 ---
#                 print(f"  步骤 4: 值提取至点...")
#                 ExtractMultiValuesToPoints(
#                     line_points, [[kriging_raster, "krig_val"]], "NONE"
#                 )

#                 # --- 步骤 5: 分类 (改进 2) ---
#                 print(f"  步骤 5: 分类点 (方法: {self.classification_method})...")
#                 vals = []
#                 with arcpy.da.SearchCursor(line_points, ["krig_val"]) as cursor:
#                     for row in cursor:
#                         if row[0] is not None:
#                             vals.append(row[0])
                
#                 if not vals:
#                     raise Exception("航线点 'krig_val' 中没有有效值。")

#                 # --- 改进 2: 灵活分类逻辑 ---
#                 if self.classification_method == "quantile":
#                     vals.sort()
#                     p33 = vals[int(len(vals) * 0.33)]
#                     p66 = vals[int(len(vals) * 0.66)]
#                     print(f"    阈值 (分位数): < {p33:.2f} (cold), < {p66:.2f} (normal)")
                
#                 elif self.classification_method == "std_dev":
#                     if np is None:
#                         raise Exception("Numpy 库未加载, 'std_dev' 方法不可用。")
#                     mean = np.mean(vals)
#                     std = np.std(vals)
#                     factor = self.classification_std_dev_factor
#                     p33 = mean - (factor * std)
#                     p66 = mean + (factor * std)
#                     print(f"    阈值 (标准差 ±{factor}): < {p33:.2f} (cold), < {p66:.2f} (normal)")
                
#                 elif self.classification_method == "manual":
#                     if not self.classification_thresholds:
#                         raise Exception("'manual' 方法需要提供 'classification_thresholds' 参数。")
#                     p33 = self.classification_thresholds["p33"]
#                     p66 = self.classification_thresholds["p66"]
#                     print(f"    阈值 (手动): < {p33} (cold), < {p66} (normal)")
                
#                 else:
#                     raise Exception(f"未知的分类方法: {self.classification_method}")
                
#                 # --- 应用分类 ---
#                 arcpy.management.AddField(line_points, "class", "TEXT")
#                 with arcpy.da.UpdateCursor(line_points, ["krig_val", "class"]) as cursor:
#                     for row in cursor:
#                         val = row[0]
#                         if val is None: row[1] = "NoData"
#                         elif val < p33: row[1] = "coldPoint"
#                         elif val < p66: row[1] = "normalPoint"
#                         else: row[1] = "hotPoint"
#                         cursor.updateRow(row)

#                 # --- 步骤 5b: 投影 (为了精确距离) ---
#                 print(f"  步骤 5b: 投影航线点至 {self.utm_sr.name}...")
#                 arcpy.management.Project(
#                     line_points, projected_line_points, self.utm_sr
#                 )
#                 print(f"    输出: {projected_line_points}")

#                 # --- 步骤 6: 执行采样 (改进 3) ---
#                 print(f"  步骤 6: 采样点 (首选距离 {self.min_distance}m)...")
#                 all_sample_ids_for_this_region = []
                
#                 for cls, num in self.sample_config.items():
#                     print(f"    正在采样 {num} 个 '{cls}' 点...")
#                     sql_where = f"class = '{cls}'"
                    
#                     # 调用新的带回退策略的采样方法
#                     sample_ids = self._sample_with_min_distance(
#                         projected_line_points, # 从投影后的图层采样
#                         sql_where, 
#                         num
#                     )
#                     all_sample_ids_for_this_region.extend(sample_ids)

#                 # --- 步骤 7: 导出合并后的采样点 ---
#                 if not all_sample_ids_for_this_region:
#                     print(f"  !! 警告: 区域 {region_id} 未能采样到任何点。!!")
#                     continue

#                 print(f"  步骤 7: 导出合并后的采样点...")
#                 where_sql_merged = f"OBJECTID IN ({','.join(map(str, all_sample_ids_for_this_region))})"
                
#                 arcpy.management.MakeFeatureLayer(projected_line_points, "temp_merged_layer", where_sql_merged)
#                 arcpy.management.CopyFeatures("temp_merged_layer", merged_output_fc)
#                 arcpy.management.Delete("temp_merged_layer")
                
#                 print(f"    合并采样点输出: {merged_output_fc} (共 {len(all_sample_ids_for_this_region)} 个点)")

#                 # --- 步骤 8: 应用符号系统 (模板分身版) ---
#                 try:
#                     if self.symbology_template_lyrx and arcpy.Exists(self.symbology_template_lyrx):
#                         print(f"  步骤 8: 应用符号系统...")
                        
#                         # 1. 强制刷新 (保留)
#                         import time
#                         import shutil # 引入文件复制模块
#                         time.sleep(1.0)
#                         arcpy.ClearWorkspaceCache_management()
                        
#                         # 2. 路径准备
#                         output_lyrx = os.path.join(
#                             os.path.dirname(self.workspace), 
#                             f"{merged_output_fc}.lyrx"
#                         )
#                         full_input_data_path = os.path.join(self.workspace, merged_output_fc)
                        
#                         # ==========================================================
#                         # !! 核心修复：创建模板副本 !!
#                         # 为了防止模板文件被锁定或产生“自我引用”冲突，
#                         # 我们为当前区域复制一个临时的、独占的模板文件。
#                         # ==========================================================
#                         temp_template_path = os.path.join(
#                             os.path.dirname(self.workspace), 
#                             f"Temp_Template_{region_id}.lyrx" # 名字带 ID，绝对唯一
#                         )
                        
#                         # 复制模板 (如果存在旧的先删除)
#                         if os.path.exists(temp_template_path):
#                             os.remove(temp_template_path)
#                         shutil.copy2(self.symbology_template_lyrx, temp_template_path)
#                         # ==========================================================

#                         # 3. 临时图层名 (加后缀 _Layer)
#                         layer_name_in_pro = f"{merged_output_fc}_Layer"
                        
#                         # 4. 清理内存旧图层
#                         if arcpy.Exists(layer_name_in_pro):
#                             arcpy.management.Delete(layer_name_in_pro)
                        
#                         # 5. 创建图层
#                         arcpy.management.MakeFeatureLayer(full_input_data_path, layer_name_in_pro)
                        
#                         # 6. 应用符号系统 (使用我们刚刚复制的 temp_template_path)
#                         # 不用字段映射，让它自动匹配
#                         arcpy.management.ApplySymbologyFromLayer(
#                             in_layer=layer_name_in_pro, 
#                             in_symbology_layer=temp_template_path
#                         )
                        
#                         # 7. 保存
#                         arcpy.management.SaveToLayerFile(layer_name_in_pro, output_lyrx, "ABSOLUTE")
                        
#                         # 8. 清理
#                         arcpy.management.Delete(layer_name_in_pro)     # 删除内存图层
#                         if os.path.exists(temp_template_path):       # 删除临时模板文件
#                             os.remove(temp_template_path)
                        
#                         print(f"    成功! 已保存带样式的 .lyrx 文件到: {output_lyrx}")
                    
#                     elif self.symbology_template_lyrx:
#                         print(f"  步骤 8: 跳过 (模板文件未找到)")
                
#                 except Exception as sym_e:
#                     print(f"    !! 警告: 样式应用失败，但数据已成功生成。原因: {sym_e}")
#                     # 尝试清理临时文件
#                     try:
#                         if 'temp_template_path' in locals() and os.path.exists(temp_template_path):
#                             os.remove(temp_template_path)
#                     except:
#                         pass
                
#                 # --- 区域完成 ---
#                 print("\n" + "*"*40)
#                 print(f"区域 {region_id} 处理成功!")
#                 print("*"*40)

#             except Exception as region_error:
#                 # 捕获此区域处理过程中的任何错误
#                 print(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#                 print(f"  !! 区域 {region_id} 处理失败: {region_error}")
#                 print(f"  !! 跳过此区域，继续下一个...")
#                 print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
#                 arcpy.env.mask = ""
#                 arcpy.env.extent = ""
#                 continue

#         print("\n" + "="*50)
#         print("!! 批量处理全部完成 !!")
#         print("="*50)
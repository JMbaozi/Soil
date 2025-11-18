import arcpy
import random
import sys
import os
from arcpy.sa import *

# 这是一个可选依赖，仅在 'std_dev' 分类法中需要
try:
    import numpy as np
except ImportError:
    print("警告: Numpy 库未找到。")
    print("      'std_dev' (标准差) 分类方法将不可用。")
    np = None

class SoilSampler:
    """
    SoilSampler v2.0
    一个用于UAV能谱数据批量处理、插值、分类和采样点生成的工具类。
    
    工作流程:
    1. 克里金插值 (Kriging)
    2. 生成UAV航线 (PointsToLine)
    3. 沿线布点 (GeneratePointsAlongLines)
    4. 提取插值 (ExtractMultiValuesToPoints)
    5. 分类 (Classify - 可选方法)
    6. 投影 (Project)
    7. 带最小距离约束和回退策略的采样 (Sample)
    8. 合并导出并(可选)应用符号系统 (Export & Symbolize)
    """

    def __init__(self, 
                 workspace,
                 batch_config,
                 output_utm_sr,
                 
                 # --- 核心参数 ---
                 time_field="collection_time", 
                 value_field="total", 
                 
                 # --- 改进 1: 参数化模型 ---
                 kriging_model_str="SPHERICAL",
                 kriging_range=300,
                 line_point_distance="5 Meters",
                 
                 # --- 改进 2: 灵活分类 ---
                 classification_method="quantile",
                 classification_std_dev_factor=0.5,
                 classification_thresholds=None,
                 
                 # --- 改进 3: 采样策略 ---
                 min_distance=100,
                 sampling_fallback_distances=None,
                 sample_config={
                     "coldPoint": 5,
                     "normalPoint": 3,
                     "hotPoint": 5,
                 },
                 
                 # --- 改进 4: 符号系统 ---
                 symbology_template_lyrx=None,
                 
                 # --- 其他 ---
                 overwrite=True):
        """
        初始化土壤采样器 v2.0。
        
        参数:
        - workspace (str): ArcPy 工作空间 GDB 路径。
        - batch_config (list): 区域配置字典的列表。
        - output_utm_sr (str/int): 必须提供的输出投影坐标系 (例如 "WGS 1984 UTM Zone 50N")。
        
        - kriging_model_str (str): 克里金模型 ("SPHERICAL", "EXPONENTIAL" 等)。
        - kriging_range (float): 克里金变程。
        - line_point_distance (str): 沿线布点距离 (例如 "5 Meters")。
        
        - classification_method (str): 分类方法: "quantile" (分位数), "std_dev" (标准差), "manual" (手动)。
        - classification_std_dev_factor (float): 用于 'std_dev' 的标准差倍数 (例如 0.5 表示 Mean +/- 0.5*STD)。
        - classification_thresholds (dict): 用于 'manual' 的阈值 (例如 {"p33": 1000, "p66": 3000})。

        - min_distance (float): 采样点的首选最小距离。
        - sampling_fallback_distances (list): 回退距离列表。如果为 None, 将自动设为 [min_distance, min_distance*0.5, 0]。
        - sample_config (dict): 每个类别的采样数量。

        - symbology_template_lyrx (str): 指向 .lyrx 样式文件的路径 (唯一值渲染 'class' 字段)。
        - overwrite (bool): 是否覆盖现有输出。
        """
        self.workspace = workspace
        self.batch_config = batch_config
        self.output_utm_sr = output_utm_sr # 强制要求
        self.time_field = time_field
        self.value_field = value_field
        
        # 改进 1
        self.kriging_model_str = kriging_model_str
        self.kriging_range = kriging_range
        self.kriging_model = KrigingModelOrdinary(kriging_model_str, kriging_range)
        self.line_point_distance = line_point_distance
        
        # 改进 2
        self.classification_method = classification_method
        self.classification_std_dev_factor = classification_std_dev_factor
        self.classification_thresholds = classification_thresholds
        
        # 改进 3
        self.min_distance = min_distance
        self.sample_config = sample_config
        if sampling_fallback_distances:
            self.sampling_fallback_distances = sampling_fallback_distances
        else:
            self.sampling_fallback_distances = [self.min_distance, self.min_distance * 0.5, 0] # 默认回退策略
        
        # 改进 4
        self.symbology_template_lyrx = symbology_template_lyrx
        
        self.overwrite = overwrite
        
        # 1. 设置 ArcPy 环境
        self._setup_environment()
        
        # 2. 解析坐标系
        self.utm_sr = self._resolve_output_sr(self.output_utm_sr)
        if not self.utm_sr:
            raise Exception("无法解析输出投影坐标系 (UTM SR)。请在初始化时显式提供 'output_utm_sr' 参数。")
    
    def _setup_environment(self):
        """设置 ArcPy 环境变量。"""
        try:
            arcpy.env.workspace = self.workspace
            arcpy.env.overwriteOutput = self.overwrite
            print(f"工作空间设置为: {self.workspace}")
        except Exception as e:
            raise Exception(f"设置工作空间失败: {e}")

    def _resolve_output_sr(self, sr_input):
        """解析并验证用于距离计算的投影坐标系 (PCS)。"""
        if not sr_input:
            print("错误: 必须在初始化时提供 'output_utm_sr'。")
            return None
        try:
            sr = arcpy.SpatialReference(sr_input)
            print(f"使用用户指定的坐标系: {sr.name}")
            if sr.type != "Projected":
                print(f"警告: 您指定的坐标系 '{sr.name}' 不是投影坐标系 (PCS)。距离计算可能不准确。")
            return sr
        except Exception as e:
            print(f"错误: 无法从输入 '{sr_input}' 创建坐标系: {e}。")
            return None
    
    def _sample_with_min_distance(self, input_fc, where_clause, num_to_sample):
        """
        (改进 3: 带回退策略的采样)
        执行带最小距离约束的抽样。
        返回：选中的 OID 列表。
        """
        candidates = []
        try:
            with arcpy.da.SearchCursor(input_fc, ["OID@", "SHAPE@"], where_clause) as cursor:
                for row in cursor:
                    candidates.append(row) # (oid, geometry)
        except Exception as e:
            print(f"    错误: 搜索 {input_fc} (Where: {where_clause}) 时出错: {e}")
            return []

        if not candidates:
            print(f"    警告：类别 '{where_clause}' 没有找到任何点。")
            return []

        if len(candidates) < num_to_sample:
            print(f"    警告：类别 '{where_clause}' 点数不足 ({len(candidates)})，无法抽取 {num_to_sample} 个。")
            num_to_sample = len(candidates) # 调整目标为所有点
        
        # 随机打乱
        random.shuffle(candidates)

        final_oids = [] # 最终 OID 列表
        
        # --- 改进 3：回退循环 ---
        for dist in self.sampling_fallback_distances:
            print(f"    ...正在尝试 {dist}m 最小距离...")
            
            final_oids = []
            final_geometries = []
            
            # 迭代筛选
            for oid, geom in candidates:
                is_valid = True
                for existing_geom in final_geometries:
                    distance = geom.distanceTo(existing_geom)
                    if distance < dist:
                        is_valid = False
                        break
                
                if is_valid:
                    final_oids.append(oid)
                    final_geometries.append(geom)

                if len(final_oids) == num_to_sample:
                    break # 抽够了

            # 检查是否在此距离下成功
            if len(final_oids) == num_to_sample:
                print(f"    成功：在 {dist}m 距离下抽出 {num_to_sample} 个点。")
                return final_oids # 成功，返回
            
            # 如果没成功，循环将继续，尝试更小的距离
            print(f"    未能抽出 {num_to_sample} 个点 (只抽出 {len(final_oids)} 个)。正在尝试下一个回退距离...")

        # --- 循环结束 ---
        # 如果所有距离 (包括 0) 都尝试完，返回最后一次尝试的结果
        print(f"    警告：已尝试所有回退距离。最终返回 {len(final_oids)} 个点 (最后一次尝试使用 {dist}m 距离)。")
        return final_oids

    def run_batch(self):
        """
        执行所有区域的批量处理。
        """
        print(f"开始批量处理 {len(self.batch_config)} 个区域...")
        print(f"采样配置: {self.sample_config}")
        print(f"分类方法: {self.classification_method}")
        print(f"采样距离策略: {self.sampling_fallback_distances}")

        for region in self.batch_config:
            
            region_id = region["id"]
            points = region["points"]
            farmland = region["farmland"]
            
            print("\n" + "="*50)
            print(f"正在处理区域: {region_id}")
            print(f"  输入点: {points}")
            print(f"  农田范围: {farmland}")
            print("="*50)

            # --- 动态定义本区域的输出名称 ---
            kriging_raster = f"kriging_total_{region_id}"
            uav_line = f"uav_route_{region_id}"
            line_points = f"route_points_{region_id}"
            projected_line_points = f"route_points_UTM_{region_id}"
            merged_output_fc = f"sample_points_{region_id}"
            
            try:
                # --- 步骤 1: 克里金插值 (改进 1) ---
                print(f"  步骤 1: 克里金插值 (模型: {self.kriging_model_str}, 变程: {self.kriging_range})...")
                arcpy.env.extent = farmland
                arcpy.env.mask = farmland
                try:
                    kriging_result = Kriging(points, self.value_field, self.kriging_model)
                    kriging_result.save(kriging_raster)
                    print(f"    输出: {kriging_raster}")
                except Exception as e:
                    print(f"    !! 步骤 1 失败 (克里金插值): {e}")
                    raise 
                finally:
                    arcpy.env.mask = ""
                    arcpy.env.extent = ""

                # --- 步骤 2: 点转线 ---
                print(f"  步骤 2: 点转线...")
                arcpy.management.PointsToLine(points, uav_line, "", self.time_field)

                # --- 步骤 3: 沿线生成点 (改进 1) ---
                print(f"  步骤 3: 沿线生成点 (距离: {self.line_point_distance})...")
                arcpy.management.GeneratePointsAlongLines(
                    uav_line, line_points,
                    Point_Placement="DISTANCE", Distance=self.line_point_distance
                )

                # --- 步骤 4: 值提取至点 ---
                print(f"  步骤 4: 值提取至点...")
                ExtractMultiValuesToPoints(
                    line_points, [[kriging_raster, "krig_val"]], "NONE"
                )

                # --- 步骤 5: 分类 (改进 2) ---
                print(f"  步骤 5: 分类点 (方法: {self.classification_method})...")
                vals = []
                with arcpy.da.SearchCursor(line_points, ["krig_val"]) as cursor:
                    for row in cursor:
                        if row[0] is not None:
                            vals.append(row[0])
                
                if not vals:
                    raise Exception("航线点 'krig_val' 中没有有效值。")

                # --- 改进 2: 灵活分类逻辑 ---
                if self.classification_method == "quantile":
                    vals.sort()
                    p33 = vals[int(len(vals) * 0.33)]
                    p66 = vals[int(len(vals) * 0.66)]
                    print(f"    阈值 (分位数): < {p33:.2f} (cold), < {p66:.2f} (normal)")
                
                elif self.classification_method == "std_dev":
                    if np is None:
                        raise Exception("Numpy 库未加载, 'std_dev' 方法不可用。")
                    mean = np.mean(vals)
                    std = np.std(vals)
                    factor = self.classification_std_dev_factor
                    p33 = mean - (factor * std)
                    p66 = mean + (factor * std)
                    print(f"    阈值 (标准差 ±{factor}): < {p33:.2f} (cold), < {p66:.2f} (normal)")
                
                elif self.classification_method == "manual":
                    if not self.classification_thresholds:
                        raise Exception("'manual' 方法需要提供 'classification_thresholds' 参数。")
                    p33 = self.classification_thresholds["p33"]
                    p66 = self.classification_thresholds["p66"]
                    print(f"    阈值 (手动): < {p33} (cold), < {p66} (normal)")
                
                else:
                    raise Exception(f"未知的分类方法: {self.classification_method}")
                
                # --- 应用分类 ---
                arcpy.management.AddField(line_points, "class", "TEXT")
                with arcpy.da.UpdateCursor(line_points, ["krig_val", "class"]) as cursor:
                    for row in cursor:
                        val = row[0]
                        if val is None: row[1] = "NoData"
                        elif val < p33: row[1] = "coldPoint"
                        elif val < p66: row[1] = "normalPoint"
                        else: row[1] = "hotPoint"
                        cursor.updateRow(row)

                # --- 步骤 5b: 投影 (为了精确距离) ---
                print(f"  步骤 5b: 投影航线点至 {self.utm_sr.name}...")
                arcpy.management.Project(
                    line_points, projected_line_points, self.utm_sr
                )
                print(f"    输出: {projected_line_points}")

                # --- 步骤 6: 执行采样 (改进 3) ---
                print(f"  步骤 6: 采样点 (首选距离 {self.min_distance}m)...")
                all_sample_ids_for_this_region = []
                
                for cls, num in self.sample_config.items():
                    print(f"    正在采样 {num} 个 '{cls}' 点...")
                    sql_where = f"class = '{cls}'"
                    
                    # 调用新的带回退策略的采样方法
                    sample_ids = self._sample_with_min_distance(
                        projected_line_points, # 从投影后的图层采样
                        sql_where, 
                        num
                    )
                    all_sample_ids_for_this_region.extend(sample_ids)

                # --- 步骤 7: 导出合并后的采样点 ---
                if not all_sample_ids_for_this_region:
                    print(f"  !! 警告: 区域 {region_id} 未能采样到任何点。!!")
                    continue

                print(f"  步骤 7: 导出合并后的采样点...")
                where_sql_merged = f"OBJECTID IN ({','.join(map(str, all_sample_ids_for_this_region))})"
                
                arcpy.management.MakeFeatureLayer(projected_line_points, "temp_merged_layer", where_sql_merged)
                arcpy.management.CopyFeatures("temp_merged_layer", merged_output_fc)
                arcpy.management.Delete("temp_merged_layer")
                
                print(f"    合并采样点输出: {merged_output_fc} (共 {len(all_sample_ids_for_this_region)} 个点)")

                # --- 步骤 8: 应用符号系统 (模板分身版) ---
                try:
                    if self.symbology_template_lyrx and arcpy.Exists(self.symbology_template_lyrx):
                        print(f"  步骤 8: 应用符号系统...")
                        
                        # 1. 强制刷新 (保留)
                        import time
                        import shutil # 引入文件复制模块
                        time.sleep(1.0)
                        arcpy.ClearWorkspaceCache_management()
                        
                        # 2. 路径准备
                        output_lyrx = os.path.join(
                            os.path.dirname(self.workspace), 
                            f"{merged_output_fc}.lyrx"
                        )
                        full_input_data_path = os.path.join(self.workspace, merged_output_fc)
                        
                        # ==========================================================
                        # !! 核心修复：创建模板副本 !!
                        # 为了防止模板文件被锁定或产生“自我引用”冲突，
                        # 我们为当前区域复制一个临时的、独占的模板文件。
                        # ==========================================================
                        temp_template_path = os.path.join(
                            os.path.dirname(self.workspace), 
                            f"Temp_Template_{region_id}.lyrx" # 名字带 ID，绝对唯一
                        )
                        
                        # 复制模板 (如果存在旧的先删除)
                        if os.path.exists(temp_template_path):
                            os.remove(temp_template_path)
                        shutil.copy2(self.symbology_template_lyrx, temp_template_path)
                        # ==========================================================

                        # 3. 临时图层名 (加后缀 _Layer)
                        layer_name_in_pro = f"{merged_output_fc}_Layer"
                        
                        # 4. 清理内存旧图层
                        if arcpy.Exists(layer_name_in_pro):
                            arcpy.management.Delete(layer_name_in_pro)
                        
                        # 5. 创建图层
                        arcpy.management.MakeFeatureLayer(full_input_data_path, layer_name_in_pro)
                        
                        # 6. 应用符号系统 (使用我们刚刚复制的 temp_template_path)
                        # 不用字段映射，让它自动匹配
                        arcpy.management.ApplySymbologyFromLayer(
                            in_layer=layer_name_in_pro, 
                            in_symbology_layer=temp_template_path
                        )
                        
                        # 7. 保存
                        arcpy.management.SaveToLayerFile(layer_name_in_pro, output_lyrx, "ABSOLUTE")
                        
                        # 8. 清理
                        arcpy.management.Delete(layer_name_in_pro)     # 删除内存图层
                        if os.path.exists(temp_template_path):       # 删除临时模板文件
                            os.remove(temp_template_path)
                        
                        print(f"    成功! 已保存带样式的 .lyrx 文件到: {output_lyrx}")
                    
                    elif self.symbology_template_lyrx:
                        print(f"  步骤 8: 跳过 (模板文件未找到)")
                
                except Exception as sym_e:
                    print(f"    !! 警告: 样式应用失败，但数据已成功生成。原因: {sym_e}")
                    # 尝试清理临时文件
                    try:
                        if 'temp_template_path' in locals() and os.path.exists(temp_template_path):
                            os.remove(temp_template_path)
                    except:
                        pass
                
                # --- 区域完成 ---
                print("\n" + "*"*40)
                print(f"区域 {region_id} 处理成功!")
                print("*"*40)

            except Exception as region_error:
                # 捕获此区域处理过程中的任何错误
                print(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"  !! 区域 {region_id} 处理失败: {region_error}")
                print(f"  !! 跳过此区域，继续下一个...")
                print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                arcpy.env.mask = ""
                arcpy.env.extent = ""
                continue

        print("\n" + "="*50)
        print("!! 批量处理全部完成 !!")
        print("="*50)
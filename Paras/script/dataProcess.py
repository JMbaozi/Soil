import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # 用于将字符串转换为字典


# ===== 设置中文字体 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SpectrumProcessor:
    # 计算光谱数据的平均 counts 并合并经纬度相同的点 计算总计数，保存为新 CSV
    # 该类已被 SpectrumAnalyzer 取代，保留以防需要简单处理时使用
    def __init__(self, input_file=None, output_file=None):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        if input_file:
            print(f"✅ SpectrumProcessor 初始化：{input_file} → {output_file}")

    # ===== 读取 CSV =====
    def load_data(self):
        print(f"📂 正在读取文件：{self.input_file}")
        self.df = pd.read_csv(self.input_file)
        print(f"✅ 已读取 {len(self.df)} 条记录")

    # ===== 平均 counts，并四舍五入 =====
    def _avg_counts(self, counts_list):
        arrs = []
        for c in counts_list:
            if isinstance(c, str):
                arrs.append(np.array(eval(c)))
        if not arrs:
            raise ValueError("⚠️ counts 数据为空")
        lengths = [len(a) for a in arrs]
        if len(set(lengths)) != 1:
            raise ValueError(f"⚠️ counts 长度不一致: {lengths}")
        avg = np.mean(arrs, axis=0)
        return np.round(avg).astype(int)

    # ===== 数据处理 =====
    def process_data(self):
        print("⚙️ 开始处理数据...")

        # 拆 location → lat/lon（保留四位小数）
        def extract_lat_lon(loc):
            coords = eval(loc)['coordinates']
            return round(coords[1], 4), round(coords[0], 4)

        self.df[['lat', 'lon']] = self.df['location'].apply(lambda x: pd.Series(extract_lat_lon(x)))

        # 按经纬度分组
        group_cols = ['lat', 'lon']
        grouped = self.df.groupby(group_cols)

        # 合并 counts
        merged_counts = []
        merged_lat = []
        merged_lon = []
        merged_time = []  # ✅ 新增：保留 collection_time
        for (lat, lon), group in grouped:
            avg = self._avg_counts(group['counts'].tolist())
            merged_counts.append(avg)
            merged_lat.append(lat)
            merged_lon.append(lon)
            merged_time.append(group['collection_time'].iloc[0])  # ✅ 同组保留第一个时间

        # 构建结果 DataFrame，只保留四列
        result = pd.DataFrame({
            'collection_time': merged_time,  # ✅ 新增
            'lat': merged_lat,
            'lon': merged_lon,
            'counts': merged_counts
        })

        # total 列
        result['total'] = result['counts'].map(np.sum)

        # 保存
        self.df = result
        print(f"🧪 分组数量（唯一点数）: {len(result)}")
        print(f"🧩 示例 counts: {result['counts'].iloc[0][:60]}")
        print("✅ 处理完成")

    # ===== 保存 CSV（counts 保存为 list 字符串） =====
    def save_data(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df_to_save = self.df.copy()
        df_to_save['counts'] = df_to_save['counts'].apply(lambda x: str(x.tolist()))
        df_to_save.to_csv(self.output_file, index=False)
        print(f"💾 文件已保存到：{self.output_file}")

    # ===== 单文件完整流程 =====
    def process(self):
        self.load_data()
        self.process_data()
        self.save_data()


class SpectrumPeak:
    """
    根据指定通道和窗口范围计算能谱峰强度（窗口求和），支持批量多通道处理
    该类已被 SpectrumAnalyzer 取代，保留以防需要简单处理时使用
    """
    def __init__(self, counts_column='counts'):
        self.counts_column = counts_column  # CSV 中的计数列名

    def compute_peak(self, counts, channel, window):
        """
        计算指定通道附近窗口的计数和
        :param counts: 样本的计数列表
        :param channel: 中心通道
        :param window: 窗口范围（正负范围）
        :return: 窗口内计数和
        """
        start = max(0, channel - window)
        end = min(len(counts), channel + window + 1)  # python 切片是左闭右开
        return sum(counts[start:end])

    def process_csv(self, input_csv, output_csv, peaks):
        """
        处理 CSV 文件，计算峰强度，并保存新 CSV
        :param input_csv: 输入 CSV 路径
        :param output_csv: 输出 CSV 路径
        :param peaks: dict, {列名: (中心通道, 窗口范围)}
                      例如: {'K40_peak': (490, 5), 'U238_peak': (600, 7)}
        """
        df = pd.read_csv(input_csv)

        # 先把原 counts 列解析成列表，方便重复使用
        counts_list = df[self.counts_column].apply(ast.literal_eval)

        for peak_name, (channel, window) in peaks.items():
            df[peak_name] = counts_list.apply(lambda x: self.compute_peak(x, channel, window))

        df.to_csv(output_csv, index=False)
        print(f'处理完成，已保存到 {output_csv}')


import pandas as pd
import numpy as np
import os
import ast

class SpectrumAnalyzer:
    """
    光谱数据分析类：支持合并平均 counts、计算总计数与多个峰值，
    并可保留指定的原始列（如 speed、height）。
    """

    def __init__(self, input_file, output_file, peaks=None, extra_columns=None):
        """
        :param input_file: 输入 CSV 文件路径
        :param output_file: 输出 CSV 文件路径
        :param peaks: dict，形如 {'K40_peak': (490, 20), 'Bi214_peak': (200, 20)}
        :param extra_columns: list，要保留的原始列名，如 ['speed', 'height']
        """
        self.input_file = input_file
        self.output_file = output_file
        self.peaks = peaks or {}
        self.extra_columns = extra_columns or []
        self.df = None
        print(f"✅ 初始化：{input_file} → {output_file}")

    # ===== 读取 CSV =====
    def load_data(self):
        print(f"📂 正在读取文件：{self.input_file}")
        self.df = pd.read_csv(self.input_file)
        print(f"✅ 已读取 {len(self.df)} 条记录")

    # ===== 平均 counts =====
    def _avg_counts(self, counts_list):
        arrs = []
        for c in counts_list:
            if isinstance(c, str):
                arrs.append(np.array(ast.literal_eval(c)))
        if not arrs:
            raise ValueError("⚠️ counts 数据为空")
        lengths = [len(a) for a in arrs]
        if len(set(lengths)) != 1:
            raise ValueError(f"⚠️ counts 长度不一致: {lengths}")
        avg = np.mean(arrs, axis=0)
        return np.round(avg).astype(int)

    # ===== 经纬度提取 =====
    def _extract_lat_lon(self, loc):
        coords = eval(loc)['coordinates']
        return round(coords[1], 6), round(coords[0], 6)

    # ===== 峰值求和函数 =====
    def _compute_peak(self, counts, channel, window):
        start = max(0, channel - window)
        end = min(len(counts), channel + window + 1)
        return int(np.sum(counts[start:end]))

    # ===== 数据处理 =====
    def process_data(self):
        print("⚙️ 开始处理数据...")

        # 提取经纬度
        self.df[['lat', 'lon']] = self.df['location'].apply(lambda x: pd.Series(self._extract_lat_lon(x)))

        # 分组字段
        group_fields = ['lat', 'lon']
        grouped = self.df.groupby(group_fields)

        merged_records = []

        for (lat, lon), group in grouped:
            record = {
                'lat': lat,
                'lon': lon,
                'collection_time': group['collection_time'].iloc[0],
                'counts': self._avg_counts(group['counts'].tolist())
            }

            # 额外字段（平均或首值）
            for col in self.extra_columns:
                if col in group.columns:
                    # 如果是数值型则求平均，否则取第一个
                    if np.issubdtype(group[col].dtype, np.number):
                        record[col] = group[col].mean()
                    else:
                        record[col] = group[col].iloc[0]
                else:
                    record[col] = np.nan  # 缺失字段填充空值

            merged_records.append(record)

        result = pd.DataFrame(merged_records)

        # ===== 计算总计数 =====
        result['total'] = result['counts'].map(np.sum)

        # ===== 计算多个峰值 =====
        for peak_name, (channel, window) in self.peaks.items():
            result[peak_name] = result['counts'].map(lambda c: self._compute_peak(c, channel, window))
            print(f"📈 已计算峰值列：{peak_name}（中心 {channel} ± {window}）")

        self.df = result
        print(f"✅ 数据处理完成，共 {len(result)} 个唯一点")

    # ===== 保存 =====
    def save_data(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df_to_save = self.df.copy()
        df_to_save['counts'] = df_to_save['counts'].apply(lambda x: str(x.tolist()))
        df_to_save.to_csv(self.output_file, index=False)
        print(f"💾 文件已保存到：{self.output_file}")

    # ===== 一键执行完整流程 =====
    def run(self):
        self.load_data()
        self.process_data()
        self.save_data()


class SpectrumAnalyzerBatch:
    """
    批量光谱分析器：支持手动指定多个文件输入输出路径
    """

    def __init__(self, file_pairs, peaks, extra_columns=None):
        """
        :param file_pairs: list，每个元素为 (input_file, output_file)
        :param peaks: dict，峰值定义，如 {'K40_peak': (490, 20), 'Bi214_peak': (200, 20)}
        :param extra_columns: list，要保留的原始列，如 ['speed', 'height']
        """
        self.file_pairs = file_pairs
        self.peaks = peaks
        self.extra_columns = extra_columns or []

    def run_all(self):
        print("🚀 开始批量处理...")
        success, failed = 0, 0

        for i, (input_file, output_file) in enumerate(self.file_pairs, start=1):
            print(f"\n📄 [{i}/{len(self.file_pairs)}] 处理文件：{os.path.basename(input_file)}")
            try:
                analyzer = SpectrumAnalyzer(
                    input_file, output_file,
                    peaks=self.peaks,
                    extra_columns=self.extra_columns
                )
                analyzer.run()
                success += 1
            except Exception as e:
                failed += 1
                print(f"❌ 文件 {input_file} 处理失败：{e}")

        print(f"\n✅ 批量处理完成！成功 {success} 个，失败 {failed} 个。")


class SoilDataExtractor:
    """
    土壤实测数据提取器
    支持批量处理多个 CSV 文件，只保留指定列并解析经纬度。
    """

    def __init__(self, file_pairs):
        """
        :param file_pairs: list，每个元素为 (input_file, output_file)
        """
        self.file_pairs = file_pairs

    def extract_fields(self, input_file, output_file):
        """提取单个文件的关键字段并保存"""
        print(f"📂 正在处理文件：{input_file}")

        df = pd.read_csv(input_file)

        # 检查字段是否存在
        required_cols = ['depth_cm', 'sample_time', 'location', 'pH', 'AP', 'NH₄⁺-N', 'AK']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ 文件 {input_file} 缺少列：{missing}")

        # 解析 location -> lat/lon
        def parse_location(loc_str):
            try:
                loc = ast.literal_eval(loc_str)
                coords = loc.get('coordinates', [None, None])
                return pd.Series({'lon': coords[0], 'lat': coords[1]})
            except Exception:
                return pd.Series({'lon': None, 'lat': None})

        df[['lon', 'lat']] = df['location'].apply(parse_location)

        # 选取需要的列
        df_out = df[['depth_cm', 'sample_time', 'lat', 'lon', 'pH', 'AP', 'NH₄⁺-N', 'AK']].copy()

        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_out.to_csv(output_file, index=False)
        print(f"✅ 已保存到：{output_file}（共 {len(df_out)} 条记录）")

    def run_all(self):
        """批量运行"""
        print("🚀 开始批量提取土壤实测数据...")
        success, failed = 0, 0

        for i, (input_file, output_file) in enumerate(self.file_pairs, start=1):
            print(f"\n[{i}/{len(self.file_pairs)}] 处理 {os.path.basename(input_file)}")
            try:
                self.extract_fields(input_file, output_file)
                success += 1
            except Exception as e:
                print(f"❌ 出错：{e}")
                failed += 1

        print(f"\n🏁 批量处理完成：成功 {success} 个，失败 {failed} 个。")


class SpeedTotalAnalyzer:
    def __init__(self, input_file, output_file):
        """
        单文件分析器
        :param input_file: 输入 CSV，需包含 'speed' 和 'total' 列
        :param output_file: 输出图像路径（含文件名）
        """
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """读取数据"""
        df = pd.read_csv(self.input_file)
        if 'speed' not in df.columns or 'total' not in df.columns:
            raise ValueError(f"⚠️ 文件 {self.input_file} 缺少 'speed' 或 'total' 列")
        self.df = df.sort_values(by='speed')
        print(f"✅ 已加载 {os.path.basename(self.input_file)}，共 {len(df)} 条记录")

    def plot_total_scatter_with_means(self):
        """绘制不同速度下 total 的散点图 + 平均值线"""
        grouped = self.df.groupby('speed')

        plt.figure(figsize=(9, 6))
        for speed, group in grouped:
            plt.scatter(
                [speed] * len(group),
                group['total'],
                alpha=0.6,
                label=f'Speed={speed}'
            )

        # 平均值折线
        means = grouped['total'].mean().sort_index()
        plt.plot(means.index, means.values, color='black', linewidth=2.5, marker='o', label='Mean Total')

        plt.title(f'Total Counts by Speed\n({os.path.basename(self.input_file)})')
        plt.xlabel('Speed')
        plt.ylabel('Total Counts')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(title='Speed', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        plt.savefig(self.output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 已保存图像至：{self.output_file}")

    def run(self):
        """执行完整流程"""
        self.load_data()
        self.plot_total_scatter_with_means()


class SpeedTotalBatchAnalyzer:
    def __init__(self, file_pairs):
        """
        批量分析器
        :param file_pairs: 列表，格式为 [(input_file, output_file), ...]
        """
        self.file_pairs = file_pairs

    def run_all(self):
        print("🚀 开始批量绘制速度-Total关系图...")
        success, failed = 0, 0

        for i, (input_file, output_file) in enumerate(self.file_pairs, start=1):
            print(f"\n📄 [{i}/{len(self.file_pairs)}] 正在处理：{os.path.basename(input_file)}")
            try:
                analyzer = SpeedTotalAnalyzer(input_file, output_file)
                analyzer.run()
                success += 1
            except Exception as e:
                failed += 1
                print(f"❌ 文件 {input_file} 处理失败：{e}")

        print(f"\n✅ 批量绘制完成！成功 {success} 个，失败 {failed} 个。")
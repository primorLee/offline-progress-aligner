# 离线进度对齐器

用途：**输入同一任务的人手、机器人两段视频的交互特征，匹配动作进度相近的帧，并筛选较可靠的对应关系。**

这次公开的是已经训练过的第 **8308 步**离线对齐器，包括代码、权重、运行示例和训练目标。模型共 182,657 个参数，权重约 0.70 MiB，CPU 就能推理。

![共享时序表示、单调匹配与同进度帧筛选](assets/overview.svg)

*离线推理示意图。视频插画表示上游来源，本仓库实际读取缓存 token；共享适配器将每帧 token 集合汇总为单位范数的 128 维表示。DTW 读取完整序列，三个条件全部通过才保留对应。画面和矩阵颜色均为示意，保留标记不等于人工确认的语义正确标签。*

[可编辑 SVG](assets/overview.svg) · [矢量 PDF](assets/overview.pdf) · [PNG 预览](assets/overview.png) · [图形来源及修正记录](assets/hero-prompt.md)

## 怎么工作

1. 上游视觉编码器先为两段视频分别提取每帧的 **512 维对象对交互 token**。每帧可以有多个对象对。
2. 共享的小型时序网络汇总对象对、相邻帧变化和短时历史，得到每帧 128 维表示。
3. DTW 在两段完整序列中找单调对应路径，允许动作快慢、停顿及多帧对应一帧。
4. 按循环返回误差、相似度和远处候选的差距筛选，输出对应帧、各自时间戳和通过标记。

**本仓库直接接收特征，不接收 MP4。** 上游视频编码器没有打包在这里。维度都是 512 不代表特征空间相同；已有权重需要匹配的编码器特征。可在自己的特征上重新训练适配器。

## 安装和运行

```bash
git clone https://github.com/primorLee/offline-progress-aligner.git
cd offline-progress-aligner
python -m pip install -e ".[test]"
python -m offline_progress_aligner align --human human.npz --robot robot.npz --weights weights/aligner_step008308.safetensors --output output/alignment.json
```

输入格式见 [FEATURE_FORMAT.md](docs/FEATURE_FORMAT.md)。输出 JSON 和 CSV，保留全部候选，并用 `accepted` 标出通过筛选的对应。`matched_human_seconds` 是映射到的人手时间；`robot_seconds` 是机器人自己的时间，两者不用等长。

默认条件：循环返回误差不超过 2 个采样机器人帧、余弦相似度至少 0.5、比远处候选高至少 0.01。比较远处候选时排除匹配位置附近 3 个采样人手帧。**这些阈值是筛选规则，不是已经校准的“正确率”。**

没有数据也能运行 [合成示例](README.md#try-without-data)。合成数据只测试软件链路，不代表真实评测效果。

## 训练与版本

使用冻结的 512 维关系特征，只训练后面的时序适配器。目标是双向循环返回损失加 `0.3 × SmoothDTW 路径损失`。循环监督知道本侧出发帧的位置，不使用人机跨视频逐帧真值。

这份权重先在 H&R/H2R 上训练到 4000 步，再在 RH20T 上继续到 8308 步；最后一阶段实际访问 6,550 对 RH20T 训练视频。它不是整个 RH20T 数据集已经全部训完的模型。[模型记录](MODEL_CARD.md)列出数据覆盖、验证指标及哈希；[训练说明](docs/TRAINING.md)提供便携训练入口。

离线对齐需要完整机器人序列，包括后续帧，所以不能直接当作在线机器人进度器。输入应当是任务相同、起止阶段相近的视频；拿两条无关视频，DTW 仍会强制给出路径，筛选也不能保证识别所有错误。

代码按 MIT 开源；权重按 CC BY-NC 4.0 提供，保留 RH20T 的非商用要求。没有公开训练视频、特征缓存、服务器配置或其他项目权重。

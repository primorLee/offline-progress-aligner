# Main figure generation record

Created and revised in the ChatGPT web image interface. The requested backend was GPT Image 2.5; the interface did not expose a verifiable image-model version. Depicted video frames and matrix colors are conceptual illustrations, not experimental images or measured similarities. This is a raster illustration, not an editable vector diagram.

The revision below was applied to an initial four-stage overview. Input format, shared adapter, embedding dimensions, bidirectional monotonic DTW and filtering criteria are grounded in the released implementation.

## Submitted revision prompt

请把刚才那张主图重新设计为 ICLR / NeurIPS / CVPR 论文中可以直接排版的 method overview，并直接生成图片。用户要求顶会风格。不要海报式大标题、营销卡片或装饰性图标。以计算图为主体：两个水平泳道分别表示人手示范和机器人视频，每条用 3 个小型简洁桌面操作示意帧表示，明确两条时间轴长度和速度不同；这些仅为概念示意，不是实验截图。两条泳道各自进入 precomputed 512-D interaction tokens（输入表示），再通过同一个 shared temporal adapter fθ（共享参数的小模块），各输出 128-D temporal embeddings。两条 embedding 汇入一个 similarity matrix，矩阵两轴标 Human time、Robot time，绘制一条单调 DTW 路径；由该匹配分出 reverse mapping，进入 correspondence filtering，短标签仅为 Cycle consistency / Similarity / Margin。最右侧输出几组两两连接的 matched frames：绿色实线表示保留的同进度帧对，浅灰虚线表示拒绝的模糊对应。主线箭头要明确显示数据计算方向，共享权重用一条轻细虚线连接两个 adapter，不能画成两个独立模型。保留不同时间轴的语义，不要暗示相同秒数即相同进度。文字使用简洁准确英文；小面板标签可用 (a) Interaction representations (b) Temporal alignment (c) Filtered matches。图名 Offline Progress Aligner 只需小号，不要占据大面积。横向 2:1 左右宽图，纯白底，扁平矢量插图视觉，黑灰细线，海军蓝/青绿表示两个输入分支，琥珀色强调匹配路径，绿色仅用于保留的对应。精密对齐，充分留白，统一无衬线字体，少量短标签，学术张量/时间序列的可读性优先，不使用渐变、阴影、大圆角卡片、拟真 3D UI、奖章、性能数字或未经实测的曲线。不引入在线控制器、视频生成、机器人动作解码或其他本项目没有的组件。重点让人看懂：跨人机、不同速度的视频，通过共享时序表示与单调匹配，离线筛出同动作进度的帧。

## Final correction prompt

对刚生成的第 2 张图做一次精准修改，保持三个区域、双分支、插画和整体构图。①完全删除最上方巨大的 Offline Progress Aligner 标题及右侧 slogan，也删除最底下两行 slogan；把释放的空间用于计算图，不加新口号或页眉。②三个 panel 的 (a)(b)(c) 标签改为小号黑色文字，去掉横向蓝灰底色条和其他大色块底板。③修正输入的科学含义：每帧有 K_t 个对象对 token，并不是每帧仅一个 512-D 向量。把人手输入下公式改成 X_t^H ∈ R^{K_t^H × 512}，机器人输入下公式改成 X_t^R ∈ R^{K_t^R × 512}；两个 precomputed 标签后可加 “per-frame token sets”。输出仍为 Z^H ∈ R^{T_H × 128} 和 Z^R ∈ R^{T_R × 128}。fθ 内部本来包含对 token 集合的 learned pooling，不另增未经实现的编码器。④双向映射先经过三条件筛选再产生右侧保留的帧对，请用明确细箭头连通；筛选写作 Similarity / Cycle consistency / Match margin，避免像一个独立无连接的说明卡。⑤其余内容不变，平面学术图形、白底、细线、统一字体，去掉浮雕阴影和营销外观，确保缩到论文双栏通栏宽度仍清楚。直接输出修改后的图片。

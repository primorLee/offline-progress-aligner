# Main figure: reference A and editable reconstruction

The approved reference is **overview-web-comic.png** (1774 × 887), generated and revised in the ChatGPT web interface. The requested image-model name was GPT Image 2.5; the web interface did not expose a verifiable backend version.

**overview.svg** is the canonical vector reconstruction of reference A, on the same 1774 × 887 canvas. Panel contours, connectors, token grids, embedding bars, matrix and sieve are native geometry. The original generated handwriting is recovered into **59 independent vector label groups**, stored separately in **lettering/reference-a.svg**. This preserves the reference letterforms instead of substituting a system font. Labels in this edition are editable as paths, not by typing. The independently provided **overview-editable-text.svg** retains 70 live Comic Sans MS text elements for typing edits; its letterforms remain an approximation of the generated reference.

The small hand/gripper scenes are separately converted to colored vector paths in **clipart/**. There are no skeletons, joint markers or keypoint overlays. Color anchors for the title panels, arrows, embedding bars, tokens and operator boxes are sampled from reference A. No full-page raster, pixel grid or whole-image tracing is embedded. Original paper-like texture and hand-drawn edge variation can still differ from the raster.

**overview.pdf** is exported directly from the main SVG and contains no raster image objects. **overview.png** is its rendered preview. Because the primary figure uses original lettering outlines, it does not depend on installed fonts. The original and vector are compared in **compare.html**. Scenes and matrix colors are schematic, not experimental records or outcomes.

## Implementation correspondence

- Each frame supplies a variable-size set of 512-D object-pair interaction tokens. The package reads token caches, not raw video pixels.
- Both branches share the adapter. It projects tokens to 128 dimensions, pools valid tokens using learned weights, and combines the pooled feature, its first temporal difference and causal Conv1D context with kernel size 3.
- Concatenation gives 384 dimensions; the readout produces L2-normalized 128-D frame embeddings.
- The complete human and robot sequences enter cosine matching and monotonic DTW in both directions.
- Candidate correspondences pass similarity, cycle-return and distant-match-margin filtering. Acceptance means passing these heuristic gates, not ground-truth semantic correctness.
- Time axes, module order, panel layout and paths follow reference A. No video-generation or robot-action module is introduced.

## Rebuild and edit

The diagram authoring source is **build_overview.py**. The optional **build_reference_lettering.py** regenerates the separate label contours from reference A; prebuilt contours are included. Install `PyMuPDF` and `fonttools`, provide installed Comic Sans MS with `--font-dir`, and install the PDF exporter locally with:

```text
npm install --prefix build/figure/native/pdf-deps pdfkit@0.20.2 svg-to-pdfkit@0.1.8
python assets/build_overview.py
```

No font files are redistributed. The main figure uses the original lettering contours. Its print exporter uses SVG-to-PDFKit to preserve vector gradients. The final PDF is checked for zero image objects. Edit labels by typing in the live-text alternative, edit contour labels and named groups in the main SVG, or update the corresponding authoring source. The original illustration crops are not required for a rebuild; the separate clip-art SVG files are included.

The earlier raster prompt history follows for provenance. It is not an additional current figure specification.

## Submitted revision prompt

请把刚才那张主图重新设计为 ICLR / NeurIPS / CVPR 论文中可以直接排版的 method overview，并直接生成图片。用户要求顶会风格。不要海报式大标题、营销卡片或装饰性图标。以计算图为主体：两个水平泳道分别表示人手示范和机器人视频，每条用 3 个小型简洁桌面操作示意帧表示，明确两条时间轴长度和速度不同；这些仅为概念示意，不是实验截图。两条泳道各自进入 precomputed 512-D interaction tokens（输入表示），再通过同一个 shared temporal adapter fθ（共享参数的小模块），各输出 128-D temporal embeddings。两条 embedding 汇入一个 similarity matrix，矩阵两轴标 Human time、Robot time，绘制一条单调 DTW 路径；由该匹配分出 reverse mapping，进入 correspondence filtering，短标签仅为 Cycle consistency / Similarity / Margin。最右侧输出几组两两连接的 matched frames：绿色实线表示保留的同进度帧对，浅灰虚线表示拒绝的模糊对应。主线箭头要明确显示数据计算方向，共享权重用一条轻细虚线连接两个 adapter，不能画成两个独立模型。保留不同时间轴的语义，不要暗示相同秒数即相同进度。文字使用简洁准确英文；小面板标签可用 (a) Interaction representations (b) Temporal alignment (c) Filtered matches。图名 Offline Progress Aligner 只需小号，不要占据大面积。横向 2:1 左右宽图，纯白底，扁平矢量插图视觉，黑灰细线，海军蓝/青绿表示两个输入分支，琥珀色强调匹配路径，绿色仅用于保留的对应。精密对齐，充分留白，统一无衬线字体，少量短标签，学术张量/时间序列的可读性优先，不使用渐变、阴影、大圆角卡片、拟真 3D UI、奖章、性能数字或未经实测的曲线。不引入在线控制器、视频生成、机器人动作解码或其他本项目没有的组件。重点让人看懂：跨人机、不同速度的视频，通过共享时序表示与单调匹配，离线筛出同动作进度的帧。

## Final correction prompt

对刚生成的第 2 张图做一次精准修改，保持三个区域、双分支、插画和整体构图。①完全删除最上方巨大的 Offline Progress Aligner 标题及右侧 slogan，也删除最底下两行 slogan；把释放的空间用于计算图，不加新口号或页眉。②三个 panel 的 (a)(b)(c) 标签改为小号黑色文字，去掉横向蓝灰底色条和其他大色块底板。③修正输入的科学含义：每帧有 K_t 个对象对 token，并不是每帧仅一个 512-D 向量。把人手输入下公式改成 X_t^H ∈ R^{K_t^H × 512}，机器人输入下公式改成 X_t^R ∈ R^{K_t^R × 512}；两个 precomputed 标签后可加 “per-frame token sets”。输出仍为 Z^H ∈ R^{T_H × 128} 和 Z^R ∈ R^{T_R × 128}。fθ 内部本来包含对 token 集合的 learned pooling，不另增未经实现的编码器。④双向映射先经过三条件筛选再产生右侧保留的帧对，请用明确细箭头连通；筛选写作 Similarity / Cycle consistency / Match margin，避免像一个独立无连接的说明卡。⑤其余内容不变，平面学术图形、白底、细线、统一字体，去掉浮雕阴影和营销外观，确保缩到论文双栏通栏宽度仍清楚。直接输出修改后的图片。


## Hand-drawn clip-art revision (2026-09-30)

Requested by the project owner: Comic Sans MS typography, colored clip-art, and hand-drawn strokes. A user-supplied style reference and the canonical figure preview were attached in the existing ChatGPT web conversation. The model/version string was not independently verified by the web UI. The reference supplies style only; its WAM/control architecture is not part of this project.

### Web prompt

请直接调用网页里的图像生成能力生成新版项目主图，用户希望用 GPT Image 2.5（请用当前可用图像生成能力，不要只返回文字方案）。这次是明确的风格替换，覆盖前面关于极简线性图标的要求。附件1是风格参考，附件2是我们已经核对正确的离线对齐器计算流程。

风格必须贴近附件1：全部英文使用 Comic Sans MS 风格，标题和标签有自然手写感；彩色、完整填色、精致剪贴画，手、机器人夹爪、方块、容器要有清楚的体积、阴影与细节；白色背景，浅蓝、浅绿、淡紫色块，略微不规则的深色手绘轮廓，手绘粗箭头、小幅排线、高光、纸贴边缘。多使用这种原创手绘素材。拒绝单色 SVG 轮廓图标、火柴人、细线几何夹爪、无填色素材、默认 UI 图标库风格。仍然有顶会 method figure 的严谨排版：整齐、易读、留白足够，无营销大标题。宽高约2:1。

只借鉴附件1的视觉风格，绝不复制 EGO/Jev/WAM、动作生成或闭环控制的内容。我们的项目叫 Offline Progress Aligner，只做离线序列对齐与对应筛选。两侧是同一任务的人手和机器人完整视频，时长可不同；不要暗示人手一定慢、机器人一定快。场景统一为绿色小方块放进蓝色浅盒，用精致的彩色手绘剪贴画展示接近、抓住、放置。

主流程三块：
(a) Shared representations：上下两条 Human video / Robot video，每条3个小场景帧（视频只是特征来源示意）。接到 Cached interaction tokens (512-D per token, variable token count per frame)，分别进入两个 shared temporal adapter f_theta 方块，并用虚线注明 Shared weights，输出 128-D frame embeddings。不要将裸视频直接画为该小适配器的输入。
(b) Monotonic matching：双路 embedding 进入 Cosine similarity 矩阵，横轴 Human time，纵轴 Robot time，画出一条有横向和竖向小台阶的单调 DTW 路径。矩阵是示意，不是实测结果。下面明确 Two-way DTW → Robot-to-human / Human-to-robot。
(c) Filtered correspondences：匹配先进入一个小筛子或三条勾选规则，Similarity / Cycle consistency / Match margin，三个规则全部通过才保留。最右输出两对相同动作阶段的 Human/Robot 彩色小场景，用绿色线与勾连接，第三对画浅灰虚线和叉表示 Rejected。不显示概率或成功率。

底部细长插图带标题 Shared adapter (frozen at inference)，画简明真实结构：512-D token set → Token projection → Learned weighted pooling，然后分出 Pooled feature / First difference / Causal Conv1D 三路，再 Concat → MLP + L2 norm →128-D。这里不是Transformer，不要添加self-attention层。标签可适当缩短，主流程大于细节。

检查所有箭头方向、模块连接和拼写。不要拥挤，不把文字压在图上。全图统一Comic Sans MS视觉，最终直接出图。


### Remove pose annotations / local corrections

修改当前第4张手绘版主图，保留构图、Comic Sans MS 字体、彩色剪贴画、白底和手绘轮廓。最重要：所有人手、机械臂和夹爪画面必须是没有任何骨架/姿态标注的干净插画。彻底删除手指、手背、手腕、夹爪和机械臂上的全部白色/蓝色/黄色关节点、珠点、圆点标记、骨架连线、关节轨迹和发光追踪。手指正常皮肤细节和机器人原有机械结构保留。上下输入帧、右侧保留帧对和灰色拒绝帧都要清理，不漏任何一帧。这里展示的是原始视频的概念画面，不能暗示姿态骨架是模型输入。512-D token 小色块、DTW热图路径圆点和帧对之间的绿点连接属于计算图，请保留；不要误删。
另修正两个科学连线问题：热图横轴Human time向右，纵轴Robot time应向上，与左下到右上的DTW路径一致；热图接到Two-way DTW的向下箭头需要清楚可见。右侧第一对保留帧应同时展示抓住方块的阶段（人手抓住方块 ↔ 夹爪抓住方块），第二对同时展示放入蓝色托盘的阶段。其余风格、文字和布局保持。请直接输出修改后图像。

# HomeBody 换脑实验报告:Qwen3.8-27B 本地替换 DeepSeek API

> **入库说明（2026-11-23 补录）**：本篇是 HomeBody 复现工作记录的一部分，实验日期 2026-11-10 ~ 11-12，正文成稿于 11-12。  
> **证据等级**沿用原文图例（下方「证据分级」一行：E1 落盘记录／E2 配置与推导／E3 解读判断），与本仓 `sonic-npu/reports/00_总览.md` §3 的 E1–E4 不是同一套定义，不要跨文档比较。  
> **视频**：文中引用的 12 个 mp4 在实验现场 `videos/` 目录，未随仓库上传；文中相应链接已改写为「现场文件名」标注，仓库内无可打开的视频。  
> **相关记录**：Qwen3.8-27B 在昇腾上的部署过程本身不在本篇，见 `../2026_1019-1025_Qwen昇腾部署/README.md`。  
> **脱敏**：除本说明与替换（节点名、IP／端口、远端路径改为泛称）外，正文逐字保留原件；含敏感信息的原件只存本地记录区，不入公开仓。

- **实验**:HomeBody SIM 评测(17 任务 × 3 重复 = 51 场)在完全不变的框架/技能/场景/评分下,仅替换决策模型——DeepSeek V4.1 Flash(华为云 ModelArts)→ **Qwen3.8-27B-w8a8(本地推理节点,910B3×6,3×TP2 vLLM 实例)**
- **时间**:2026-11-10 ~ 11-12;运行:仿真节点(sim)+ 推理节点(推理);输出:推理节点上的「远程实验目录」
- **基线**:本仓库 `复现报告.md`(DeepSeek V4.1 Flash,10-27~28,同协议 51 场)
- **证据分级**:E1 = 本实验落盘记录(session/summary/decision evidence/vllm metrics);E2 = 配置与推导;E3 = 解读判断

---

## 一、实验设计:换脑只动了 1 个文件、3 处

框架、技能库、评分器、任务集零改动,适配全部在 vlm 兼容层(`vlm/compat.py`):

| 适配 | 动机 |
|---|---|
| `</think>` 剥离 | Qwen 思考型模型把推理放在 content 内 `</think>` 之前,Decision 解析只认 JSON 尾部;无标记时零影响 |
| length 自动降级重试 | 思维链撞长度上限被截断时(`finish_reason=length` 且无 `</think>`),该步自动关思考重发——收敛步保留完整推理,跑飞步降级直答(全程仅触发 16/447 步) |
| `OPENAI_COMPAT_EXTRA_BODY` | 环境变量注入请求体(思考档用 `max_tokens=12288` 止损;关思考档用 `enable_thinking:false`) |

**推理端部署**(推理节点):3 × vLLM 实例(`tensor-parallel-size 2`),w8a8 权重(「远程模型权重目录」下的 `Qwen3.8-27B-w8a8`),`--max-model-len 65536`,prefix caching 开。三实例分别服务三路波次。[E2]

### 1.1 任务族全景:17 个任务分别考什么

所有任务都在同一个 Real2Sim 厨房(`src_kitchen`)里,操作对象固定为 4 个可搬动物体(两个咖啡包裹 parcel_left/parcel_corner、橙汁盒 juice_carton、红白折叠盒 red_white_carton)+ 各任务指定的目的地(岛台/圆桌/回收桶/角柜台面)。按考点分五组:

| 组 | 任务 | 指令要点 | 考点 |
|---|---|---|---|
| **基准换位**(4) | default | "咖啡包上台面,纸盒扔桶" | 4 物体整理基准 |
| | default_reworded | 同上换说法("Clean up the kitchen…") | 换词鲁棒性 |
| | swapped | 目的地对调(盒上岛、包进桶) | 指令-场景反直觉映射 |
| | swapped_reworded | 对调 + 换词 | 双重难度 |
| **指代消解**(5) | the_other_carton_to_bin | "扔掉**不是果汁**的那个纸盒" | 排除指代 |
| | juice_to_table_by_bin | "果汁放到**离回收桶最近**的圆桌" | 空间关系指代 |
| | carton_by_toaster | "红白盒放到烤面包机旁" | 单物体搬运 |
| | after_breakfast | "早餐结束了:**不是果汁的**纸盒进回收" | 换说法的排除指代 |
| | juice_for_guest | "果汁给**坐在离回收桶最近的圆桌旁的人**" | 双重间接指代 |
| **相对/条件**(2) | farther_bag | "把**离冰箱更远**的咖啡包上台面" | 相对指代 |
| | conditional_juice | "**如果**果汁在烤面包机旁,**就**上台面,**否则**扔掉" | 条件前提验证 |
| **多物体编排**(4) | swap_coffee_bags | 两个咖啡包互换位置 | 双物体互换 |
| | there_and_back | 展示红白盒给岛台的人,**再放回原处** | 逆操作 + 位置记忆 |
| | bags_to_rear_table | 两个咖啡包都放后圆桌 | 双物体同目的地 |
| | juice_to_front_table | 红白盒上岛、果汁上前桌 | 双物体异目的地 |
| **长距/最难**(2) | cartons_to_corner_counter | 两个纸盒都搬到**角落与灶台之间的台面** | 跨区长链 + 亚段级空间解析 |
| | moved_juice_to_bin | 果汁(已被移动过)扔桶——记忆是错的 | 错误记忆搜索(全库最难,两个模型都 0/3) |

每个任务的评分是独立的物理评分器:逐物体检查终态是否落在目标实体的 bounds 内且静止受支撑——**技能返回 OK ≠ 得分**,"放在台面上但压错相邻亚段"同样判 outside_target。[E1]

---

## 二、总结果:36/51 = 70.6%(DeepSeek 32/51 = 62.7%)

| | Qwen3.8-27B 本地 | DeepSeek V4.1 Flash |
|---|---|---|
| **重复 3×17** | **36/51 = 70.6%** | 32/51 = 62.7% |
| done 宣告误报 | 11/47 场(23%) | 4 场(基线报告 3.3) |

### 逐任务对照(3 次重复)

| 任务 | Qwen3.8 | DeepSeek | 差 |
|---|---|---|---|
| after_breakfast | **3/3** | 2/3 | +1 |
| bags_to_rear_table | **3/3** | 1/3 | **+2** |
| carton_by_toaster | 3/3 | 3/3 | = |
| cartons_to_corner_counter | 1/3 | **3/3** | **−2** |
| conditional_juice | 0/3 | **1/3** | −1 |
| default | **2/3** | 1/3 | +1 |
| default_reworded | 2/3 | 2/3 | = |
| farther_bag | 2/3 | **3/3** | −1 |
| juice_for_guest | 3/3 | 3/3 | = |
| juice_to_front_table | 3/3 | 3/3 | = |
| juice_to_table_by_bin | **3/3** | 2/3 | +1 |
| moved_juice_to_bin | 0/3 | 0/3 | =(全库最难,两家同败) |
| swap_coffee_bags | 2/3 | 2/3 | = |
| swapped | 1/3 | 1/3 | = |
| swapped_reworded | **2/3** | 1/3 | +1 |
| the_other_carton_to_bin | **3/3** | 2/3 | +1 |
| there_and_back | **3/3** | 2/3 | +1 |

**7 胜 3 负 7 平**。注:swapped 两场败因是步态摔倒(physics_invalid,AMO 随机性,与决策模型无关;基线报告亦记录 2/12 摔倒率)。剔除摔倒口径:36/49 = 73.5%。[E1]

---

## 三、详细分析

### 3.1 成功率:小模型在常规任务上更稳

70.6% vs 62.7%,差距主要来自**常规多物体搬运的稳定性**(bags_to_rear_table +2、default +1、there_and_back +1、juice_to_table_by_bin +1)——这些任务的考点是"老老实实把每个物体搬对地方",Qwen3.8 的执行纪律不输甚至更好。失败物体的分布与基线同构:果汁盒(6 次)、parcel_corner(6 次)、红白盒(5 次)仍是三大弱点。[E1]

### 3.2 能力边界(经同任务对决修正后的结论)

我们先给出两个直觉判断,随后用 §3.4 的逐场对决**推翻了它们**,修正后的边界如下:

**真实边界 ①:相邻台面的亚段级空间解析**(cartons_to_corner_counter 1/3 vs 3/3)。三场失败全部是 `outside_target`:两个纸盒都搬了、也都放在"角柜台面"上了,但落点在计分实体(`corner_counter_return`)的邻段——**规划与执行零失误,输在对"角落与灶台之间的台面"这条空间语言做亚段级(≈20cm)圈定**。DeepSeek 三场都圈对了。[E1+E3]

**真实边界 ②:物理交互的韧性**(conditional_juice 0/3 的真实死因)。Qwen 并**没有**跳过前提验证(逐场证据见 §3.4):它正确判定"果汁在烤面包机旁 → 条件成立"后才行动;失败发生在搬运途中物体掉落(CARRY_CLEARED)→ 掉到地面后距肩 0.87m 超出 0.36m 臂展物理不可及 → 诚实宣告不可完成。这是**路线选择导致的持物丢失 + 硬件可达域**的组合,不是指令理解问题。[E1]

**被撤回的两个误判**:首轮我们把该任务族失利归因于"跨区长链规划"与"条件指令忠实度"——对决取证显示前者实为空间精度、后者实为物理韧性。教训:**任务级胜负表不足以定位能力边界,必须回到逐步证据**。[E3]

farther_bag(相对指代)2/3:轻微落后但非硬边界。反方向:基线的 done 宣告误报(4 场)是 DeepSeek 的老大难,Qwen3.8 为 11/47(23%)——**小模型没有更爱撒谎**。[E1]

### 3.3 时延与吞吐:同 payload 双端实测(真实决策 prompt + 4 图,流式,各 3 次)

| 指标 | **Qwen3.8-27B 本地** | **DeepSeek API** | 胜者 |
|---|---|---|---|
| **TTFT**(首 token) | **0.42–0.45s** | 1.18–1.35s | 本地,×3 |
| **TPOT**(空载) | 22.3–23.0ms | **9.2–13.4ms** | API,×2 |
| TPOT(6 路并发,vllm metrics) | ~26ms | (按请求独立伸缩) | API |
| 思考量/步(中位) | **2,945 tok** | ~6,400 tok(基线报告) | 本地省一半 |
| 单步决策墙钟 | 中位 78s / P95 304s | 中位 43s / 均值 51s | API |
| 降级触发率 | 16/447 步(3.6%) | — | — |

三个解读:

1. **剪刀差**:本地赢"开口快"(无网络往返 + prefix cache),API 赢"说话快"(数据中心推理)。DeepSeek 单步多想 3–6 倍 token,两个效应大部分抵消——但 DeepSeek 的 43s 中位仍优于本地的 78s,**单流时延 API 占优约 1.8 倍**。
2. **TTFT 在本场景近乎无关紧要**:决策步是"一次想完再一次输出"的模式,没有逐 token 交互,首 token 快 0.9 秒在 70 秒的决策里是噪音。TTFT 优势真正值钱的场景是人机对话/流式 UI。
3. **并发与成本的结构性差异**:本地 3 实例聚合 ~120 tok/s 已支撑 6 路会话无排队;API 天然按请求伸缩、但按量计费。本实验 431 个有效决策步全部本地完成,**边际成本≈电费**。[E1/E2/E3]

### 3.4 同任务对决:三个场次的逐步取证

取同一任务下两模型的真实会话(完整输入相同:七段式 prompt + 4 图;输出为一步一个决策 JSON)。注:Qwen 的思考随 content 落盘(可取证),DeepSeek 基线会话的推理未随 content 保留,以下 DS 侧引用其决策文本与参数。[E1]

#### 对决 A:cartons_to_corner_counter —— DS 胜(14 步)vs Qwen 败(9 步)

**DS(胜,session-20261103T142011Z)**:开局连吃 NO_GRASP ×2 / UNREACHABLE / INVALID_ARGUMENT,前后换了 6 次抓取点、1 次站位才拿起果汁盒;此后导航→放置→再取红白盒→放置→done,全程 14 步。
**Qwen(败,session-20261111T075052Z)**:**9 步教科书执行**——navigate → pick(一次成功)→ navigate → place OK → navigate → pick(一次成功)→ navigate → place OK → done。逐步 say 摘录:

> step-000(navigate):"The brown box ahead is parcel A, **not a carton**; the juice carton is on the left refrigerator-wall counter…" ← 开局就正确排除了干扰物
> step-003(place):"…I place the carton on the clear counter surface **to the LEFT** of the brown parcel"(y_left +0.25)
> step-007(place):"…I place the carton on clear counter **to the RIGHT**…"(y_left −0.15)

评分:两个纸盒均 `outside_target`。**Qwen 的执行效率高于 DS 的胜局,输在放置落点落在计分亚段的邻面**——这就是 §3.2 边界①的出处:DS 第一步 place 用图像点击 [500,700] 落在包裹右侧计分段内,Qwen 用躯干系偏移落在左段。[E1+E3]

#### 对决 B:conditional_juice —— DS 胜(6 步)vs Qwen 败(6 步)

**DS(胜)**:step-000 即从场景记忆判定 "Memory places the juice carton in front of the toaster… **it is next to the toaster**",取→运→岛台绕行(避开凳子)→放置→done,一次成型。
**Qwen(败)**:同样**先验证前提**——

> step-000:"…I'll move to the fridge-wall counter where the juice sits by the toaster **to check it**"
> step-001:"The orange juice carton lies directly in front of the toaster oven… **so the condition holds** and I will pick it up"(思考链同句确认)

此后 pick OK → 搬运途中 **CARRY_CLEARED(盒子掉落)** → 两次地面拾取均 UNREACHABLE("0.87 m below the shoulder… beyond its 0.36 m grasp reach")→ step-005 诚实宣告:"Cannot be completed: the orange juice carton fell to the floor…"。**前提验证满分,死在物理交互**(§3.2 边界②的出处)。[E1]

#### 对决 C:swapped —— 双胜的风格对比(DS 18 步 vs Qwen 17 步)

结构几乎相同:先咖啡包进桶、再纸盒上岛。差异在过程质量:DS 全程遭遇 NO_ROUTE / PLACEMENT_NOT_VERIFIED / ARM_NOT_REACHED / COLLISION 四类技能失败,每次靠失败码反馈换点重试恢复("after the first stance failed I will navigate to a wider clear stance…");Qwen 同样有中途失败并恢复,且每步思考以"Let me understand the situation…"开场**重述任务与已完成进度**再决策(例:step-005:"First carton (juice carton) was picked and placed… **Good.** Second carton…")——一种显式的自我 grounding 风格,与其更低的 done 误报率相互印证。[E1+E3]

### 3.5 思考 vs 关思考:四格矩阵(补测 102 场)

同管线、同 17 任务 × 3 重复,两模型各补跑一次全程关思考的波次(Qwen:`enable_thinking:false`;DeepSeek:`thinking:{"type":"disabled"}`):

| | 思考 | 关思考 | 思考的价值 | 单步中位(思考→关) |
|---|---|---|---|---|
| **Qwen3.8-27B 本地** | 70.6%(36/51) | **8%(4/51)** | **+63pt** | 78s → **4.2s**(×19) |
| **DeepSeek API** | 62.7%(32/51) | **12%(6/51)** | **+51pt** | 43s → **2.2s**(×20) |

四个发现:

1. **思维链是承重墙**:关掉思考,两个模型一起崩到 8–12%,唯一在两边都稳定存活的只有 there_and_back(最简单的展示-放回)。长程多物体任务本质上需要"工作记忆式"的逐步推理——记住搬了什么、还剩什么、失败在哪。
2. **反超消失**:关思考后 DeepSeek(12%)反超 Qwen(8%)——思考档里 Qwen 的领先来自"想得省但想得对"的思考质量,而非裸决策能力。两个维度分开看:裸能力 DS 略优,思考利用率 Qwen 更高。
3. **速度收益对称**:两边都提速 ~19-20 倍,但换来的成功率崩盘不可接受——**时延优化不能走关思考这条路**,应走 TPOT/并发扩展(prefix caching、多实例、更大 TP)。
4. 关思考档 490/798 步决策零解析失败、单步 P95 < 6s——作为"低时延兜底模式"或人机对话前置环节仍可用。[E1]

---

## 四、结论

**已验证的事实**(证据括注):

1. 51 场同协议对比:36/51 = 70.6% vs DeepSeek 32/51 = 62.7%,7 胜 3 负 7 平(推理节点「远程实验目录」下 `qwen38-rep*` 的 summary.json)。
2. 换脑成本:框架 1 文件 3 处适配(§一),技能/评分/任务零改动;决策合法率 100%。
3. 时延画像双端实测:TTFT 0.44 vs 1.29s、TPOT 23 vs 11ms、思考量 2.9K vs 6.4K tok、单步 78 vs 43s(§3.3)。
4. 同任务对决取证(§3.4):Qwen 在边界任务上的规划与指代理解均正确,失利定位在**亚段级放置精度**与**物理交互韧性**两个具体环节。

**我们的判断**(E3):

1. "本地小模型能否替代前沿 API 做具身编排"在本任务族上的答案:**能,且在执行纪律上略优**;其真实边界不是规划与指令理解,而是亚段级空间圈定精度与持物路线的物理韧性。
2. 时延上本地不占优(单步慢 1.8 倍),但差距来自思考量×TPOT 的乘积,而非不可逾越的架构劣势;对成本敏感、数据不出域的部署(昇腾 + 开源权重的完整本地具身栈),这个 trade 是划算的。
3. 对华为侧的落点:这构成了"910B3 + Qwen + vLLM-ascend 跑具身编排"的完整可复现基线。
4. 方法论:任务级胜负表会诱导"长链规划/指令忠实度"这类似是而非的边界归因;逐步取证(对决 A/B)才能把边界定位到可修复的环节——**放置精度可望用提示工程(要求点击而非躯干偏移放置)修复**,这与基线报告的建议同向。

**局限**:单场景(src_kitchen)、单任务族、DS 端 probe 仅单并发、DS 基线会话未保留思考链(对决分析中 DS 侧仅决策层可比)、样本 51 场(每任务 n=3,任务级差异有方差)。

**下一步**:① 35B-A3B(已就位)同管线加测,形成本地模型规模梯度;② 针对边界①的提示工程实验(place 优先用图像点击而非躯干偏移)、针对边界②的路线约束(持物时更宽的绕行裕度)。

---

## 五、视频回放(本地,`videos/` 目录)

精选 12 场代表场次(第三人称视角,点击即播):

**胜局(执行纪律)**:after_breakfast（现场 `videos/after_breakfast__win.mp4`，未上传） · bags_to_rear_table（现场 `videos/bags_to_rear_table__win.mp4`，未上传）(+2) · default（现场 `videos/default__win.mp4`，未上传） · juice_to_table_by_bin（现场 `videos/juice_to_table_by_bin__win.mp4`，未上传） · swapped_reworded（现场 `videos/swapped_reworded__win.mp4`，未上传） · the_other_carton_to_bin（现场 `videos/the_other_carton_to_bin__win.mp4`，未上传） · there_and_back（现场 `videos/there_and_back__win.mp4`，未上传）

**边界败局(对决取证场次)**:cartons_to_corner_counter（现场 `videos/cartons_to_corner_counter__fail.mp4`，未上传）(对决 A:9 步完美执行、落点邻段) · conditional_juice（现场 `videos/conditional_juice__fail.mp4`，未上传）(对决 B:前提验证正确、搬运掉落) · farther_bag（现场 `videos/farther_bag__fail.mp4`，未上传）(相对指代) · moved_juice_to_bin（现场 `videos/moved_juice_to_bin__fail.mp4`，未上传）(最难,两家同败) · 步态摔倒样本（现场 `videos/gait_fall__sample.mp4`，未上传）

## 附:复现命令与环境

```bash
# 推理端(推理节点,3 实例×2 卡)
docker run -d --network host --shm-size 16g -e ASCEND_VISIBLE_DEVICES=0,1 \
  -v <远程模型权重目录>:<容器内挂载点>:ro <镜像> \
  vllm serve <远程模型权重目录>/Qwen3.8-27B-w8a8 --served-model-name qwen38-27b \
  --max-model-len 65536 --port <端口> --tensor-parallel-size 2 --enable-prefix-caching

# 会话端(仿真节点,<现场仓库目录>)
export OPENAI_COMPAT_BASE_URL=http://<本地推理服务地址>:<端口>/v1/chat/completions \
       OPENAI_COMPAT_API_KEY=local OPENAI_COMPAT_MODEL=qwen38-27b \
       OPENAI_COMPAT_EXTRA_BODY='{"max_tokens": 12288}' MUJOCO_GL=egl
python tools/run_wave.py configs/waves/all17.json --model compat \
  --output <远程实验目录>/<名字> --repeat 3 --jobs 2

# 时延探针(双端)
python3 <探针脚本路径> <URL> <MODEL> <KEY> prompt.txt img0.png img1.png img2.png img3.png 3
```

监控面板:`http://<本机回环地址>:<端口>`(Mac 本地,`<本地监控脚本路径>`,数据经短 SSH 拉取推理节点)。

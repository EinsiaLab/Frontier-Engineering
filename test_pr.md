openai codex团队分享的在codex上使用skills系统的经验：https://developers.openai.com/blog/skills-shell-tips
 
把 Skill 写成“路由规则”，不是广告文案
 Skill 的 ‎`description` 就是模型的决策边界，要清楚回答：什么时候用、什么时候不用、输出长什么样。最好直接在描述里写一个简短的 “Use when / Don’t use when”，并且说清输入、输出和成功标准。

用「负例 + 边界情况」来减少错误触发
 仅仅暴露 Skills 可能一开始会让正确触发率下降。做法是：在说明里写清楚“不要在什么情况下调用这个 skill（以及应该改用什么）”，并补充各种 edge cases。Glean 的经验是，加了这些后，触发率从一度下降 20% 又爬回来了。

把模板和示例塞进 Skill 内部，而不是系统提示里
 所有报告模版、标准总结格式、例子，都放在 Skill 里：

Skill 被调用时，示例才会进上下文，

不相关任务不会被这些模板拖累 token。
 对知识工作尤其有用，比如结构化报告、工单升级摘要、账号规划、数据分析写作等。


从一开始就按「长跑」设计：复用容器 + 默认启用压缩
 长链路 agent 不要当成一次性 prompt：

多步任务里尽量复用同一个容器，这样依赖、缓存文件、中间结果都能保留。

调用下一步时传 ‎`previous_response_id`，让模型在同一个 thread 里接着干活。

把 compaction 当成默认机制，而不是「爆 context 时的急救」。
 这样能明显减少「重新来过」、上下文断裂等问题。


需要确定性时，直接强制让模型用某个 Skill
 默认是模型自己决定要不要调用 Skill。但如果某条生产流程有明确 contract，就直接在指令里说“Use the ‎`<skill name>` skill.”，用硬约束换取可预期的行为。

小心「Skill + 网络访问」的组合，按高风险设计安全边界
 Skill 里面可能有很强的操作能力，一旦配上开放网络，就容易变成数据外流的通道。所以：

Skill 和 Shell 默认可以用；

网络访问只在有严格 allowlist、任务范围很窄时开启；

把网络当成需要强确认的危险工具，而不是默认能力。


统一用 ‎`/mnt/data` 做 artifact 交接点
 在托管 Shell 里约定：所有要给人看、要回传给系统的文件都写到 ‎`/mnt/data`，比如报告、清洗后的数据集、最终表格等。
 心智模型是：工具写磁盘、模型在磁盘上思考、开发者从磁盘上取结果。

把网络 allowlist 当成「组织层 + 请求层」双层系统

组织层：管理员配置一个小而稳定的 org-level allowlist，代表这个组织总体信任的目标域名。

请求层：每个请求带一个更小的 ‎`network_policy`，只能是 org allowlist 的子集。
 任何超出组织 allowlist 的域名都会直接报错。


用 ‎`domain_secrets` 做鉴权，避免暴露凭证
 调用受保护 API 时，用 ‎`domain_secrets` 注入认证信息。模型看到的只是 ‎`$API_KEY` 之类的占位符，真正的 key 在 sidecar 里按目的域名注入，避免凭证在对话上下文里流动。

线上线下用同一套 API 和 Skill
 Skills 同时适配托管 Shell 和本地 Shell。Shell 也可以用本地执行模式：你自己执行 ‎`shell_call` 然后把 ‎`shell_call_output` 传回模型。开发流程可以是：


本地起步，快速迭代、用内部工具、方便 debug；

稳定后迁到托管容器，获得可复现、隔离的运行环境；

Skill 文件本身保持不变，只是换了执行环境。

再往后，文章给了三种组合模式：

Pattern A：安装依赖 → 拉数据 → 写报告到 ‎`/mnt/data`；

Pattern B：Skill 编排 + Shell 执行，拿稳定可重复的 workflow；

Pattern C：用 Skill 携带企业级 SOP，把多工具编排的复杂流程封装起来，提高准确率、缩短首 token 延迟。

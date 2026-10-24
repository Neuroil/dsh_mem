/**
 * 确定性合成语料（benchmark 专用）：
 *
 * 设计目标——让插件自带的 deterministic 词袋嵌入产生**有结构的近邻关系**：
 * 同族文本共享大量 token → 哈希桶重叠 → 族内余弦显著高于族间。这样：
 *   - 检索 benchmark 的近邻/CSLS 阶段在非退化分布上计时；
 *   - 睡眠周期 benchmark 的聚类（clusterCos 门槛）能真正出簇。
 * 语料覆盖中英双语、短句偏好与长句决策，长度分布贴近真实记忆。
 * 全部由种子 PRNG（mulberry32）驱动：同种子 → 同语料，跨机可复现。
 */

/** mulberry32：32 位种子 → [0,1) 均匀流。 */
export function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)]

/** 主题族：共享词汇 + 模板。`{x}` 类槽位由 RNG 填充（引入唯一性熵，不破坏族内 token 重叠）。 */
const FAMILIES = [
  {
    key: 'database',
    templates: [
      'The team standardized all new services on PostgreSQL as the primary database in {year}.',
      'Database schema migrations must go through the review pipeline before touching production PostgreSQL.',
      'PostgreSQL replication lag is checked every morning as part of the database health routine.',
      'The analytics workload runs on a read replica of PostgreSQL so heavy query load never hits the primary database.',
      'User prefers PostgreSQL over MySQL for anything involving JSON columns and recursive query patterns.',
      'A database index on the events table was added after the PostgreSQL query plan showed a sequential scan.',
    ],
  },
  {
    key: 'editor',
    templates: [
      'User edits code in Neovim with a custom keymap built over {years} years of tweaking.',
      'The Neovim config lives in a dotfiles repository and every keymap change is committed with a reason.',
      'User dislikes IDEs with heavy GUI and prefers terminal Neovim even for large refactoring sessions.',
      'Neovim plugins are pinned to exact versions because a plugin update once broke the editor for a full day.',
      'User pairs Neovim with tmux and keeps a keymap cheat sheet taped to the monitor.',
    ],
  },
  {
    key: 'coffee',
    templates: [
      'User drinks pour-over coffee every morning and grinds the beans fresh right before brewing.',
      'The preferred coffee roast is medium-light from a local roaster near {city}.',
      'User refuses espresso after 3pm because caffeine past that hour ruins sleep quality.',
      'A gooseneck kettle and a scale are non-negotiable parts of the user morning coffee ritual.',
      'User keeps a brewing journal noting coffee dose, water temperature and extraction time.',
    ],
  },
  {
    key: 'deploy',
    templates: [
      'Deployment happens through a blue-green pipeline and Fridays are a hard freeze window.',
      'The deploy checklist requires a second reviewer for any change touching the payment service.',
      'Rollbacks are rehearsed monthly so the deploy runbook stays truthful under pressure.',
      'Infrastructure changes go through Terraform plans reviewed by the platform team before deploy.',
      'A failed deploy in {month} led to the rule that every migration must ship with a tested rollback.',
    ],
  },
  {
    key: 'python',
    templates: [
      'User writes Python with strict type hints enforced by mypy in the CI pipeline.',
      'Data analysis is done in Python notebooks first, then promoted to a tested module in the repo.',
      'User prefers dataclasses over dictionaries for any Python struct that survives one screen of code.',
      'Python dependency management on this project uses uv with a lockfile committed to the repository.',
      'A recurring Python habit: write the failing pytest case first, then the implementation.',
    ],
  },
  {
    key: 'running',
    templates: [
      'User runs {km} kilometers every Tuesday and Thursday before work.',
      'Marathon training blocks are planned 16 weeks ahead with a hard cap on weekly mileage growth.',
      'User tracks resting heart rate as an overtraining signal and skips a run when it spikes.',
      'Race nutrition was settled after the {city} marathon: gels every 40 minutes, nothing new on race day.',
      'Running shoes are replaced every 700 kilometers because the user got shin splints ignoring this once.',
    ],
  },
  {
    key: 'finance',
    templates: [
      'The household budget is reviewed on the last Sunday of each month over tea.',
      'User keeps an emergency fund covering 8 months of expenses in a separate savings account.',
      'Any purchase above {amount} euros requires a 48-hour cooling-off period before buying.',
      'Index funds beat stock picking for this user after a costly lesson with a single stock in {year}.',
      'Subscriptions are audited twice a year; anything unused for two billing cycles gets cancelled.',
    ],
  },
  {
    key: 'gaming',
    templates: [
      'User plays strategy games on weekends but avoids anything with daily login rewards.',
      'The gaming setup uses a 27-inch 1440p monitor because 4K felt wasteful for the GPU budget.',
      'User finishes a game or drops it for good — no graveyard of half-played saves.',
      'Co-op sessions with {name} happen on Sunday evenings, usually strategy or co-op roguelikes.',
      'A gaming rule: no microtransactions, even cosmetic, after the loot box incident in {game}.',
    ],
  },
  {
    key: 'chinese-food',
    templates: [
      '用户每周五晚上固定去杭州公司楼下的那家衢州菜馆吃小炒。',
      '用户不吃香菜，点外卖时每次都会在备注里写明去掉香菜。',
      '用户早餐偏好豆浆油条，周末会自己做葱油饼配粥。',
      '用户觉得公司楼下的咖啡太贵，宁愿自己手冲也不买。',
      '用户对辣度的容忍是中辣，特辣会胃疼，这个界限在 {year} 年吃火锅时确认过。',
    ],
  },
  {
    key: 'chinese-family',
    templates: [
      '用户和父母每周日晚上通一次视频电话，这是雷打不动的安排。',
      '用户的女儿在学钢琴，每天练二十分钟，用户负责陪练和录音。',
      '用户的妻子在杭州一家医院做护士，夜班日的晚饭由用户负责。',
      '用户家的装修风格是日式原木风，家具大部分是白橡木。',
      '用户计划 {year} 年春节带全家去哈尔滨看冰雕。',
    ],
  },
  {
    key: 'chinese-work',
    templates: [
      '用户的团队每季度做一次线上故障复盘，报告要写成无责备的样式。',
      '用户公司的大老板姓陈，喜欢在周会上引用《毛选》讲战略。',
      '用户的绩效沟通每半年一次，上次的结果是超出预期。',
      '用户在推进把核心服务从虚拟机迁移到 Kubernetes 的事情，阻力主要在安全团队。',
      '用户每周三下午是免会议日，用来写代码和做代码评审。',
    ],
  },
  {
    key: 'books',
    templates: [
      'User reads physical books before bed and keeps a yearly reading log in a paper notebook.',
      'Science fiction is the comfort genre; {name} series got re-read three times during hard weeks.',
      'User annotates margins sparingly — only arguments worth disagreeing with in writing.',
      'The local library holds a monthly used-book sale where user caps spending at {amount} euros.',
      'A reading rule from {year}: finish or abandon within two weeks, no guilt either way.',
    ],
  },
]

const YEARS = Array.from({ length: 41 }, (_, i) => String(1990 + i))
const CITIES = ['Hangzhou', 'Berlin', 'Lisbon', 'Kyoto', 'Toronto', 'Chengdu', 'Shenzhen', 'Osaka', 'Austin', 'Zurich', 'Qingdao', 'Melbourne']
const NAMES = ['Lin Tao', 'Mara', 'Kenji', 'Priya', 'Old Chen', 'Xiao Wang', 'Diego', 'Iris', 'Ravi', 'Emma', 'Zhou Jie', 'Karl']
const AMOUNTS = ['200', '500', '1000', '2000', '3000', '5000']
const MONTHS = ['January', 'March', 'April', 'July', 'September', 'November']
const GAMES = ['Starfall', 'Ironclad', 'Neon Abyss', 'Dustfront', 'Emberwind']
const KMS = ['5', '8', '10', '12', '15']
const VARY_YEARS = ['three', 'five', 'eight', 'ten']

/** 每族带槽位的尾巴从句：引入唯一性熵的同时复用族词汇（不稀释族内聚类信号）。 */
const TAILS = {
  database: ['This was reaffirmed in the {month} database review.', 'The decision came up again in {year} during migration planning.'],
  editor: ['The setup was rebuilt from scratch in {year}.', 'This came up again during the {month} dotfiles cleanup.'],
  coffee: ['The routine has held since {year}.', 'This was revisited during the {month} trip to {city}.'],
  deploy: ['The rule was written down after the {month} incident review.', 'It came up again in the {year} on-call rotation.'],
  python: ['The convention was agreed in the {month} code review.', 'It was revisited in {year} during the big refactor.'],
  running: ['This plan was adjusted in {year} after a race in {city}.', 'The rule survived the {month} training block.'],
  finance: ['The policy was reviewed in the {month} budget meeting.', 'It came up again in {year} during tax season.'],
  gaming: ['The rule was set in {year} and has held since.', 'It came up during the {month} session with {name}.'],
  'chinese-food': ['这个习惯从 {year} 年就定下来了。', '这件事 {month} 那次聚餐时又确认过。'],
  'chinese-family': ['这个安排从 {year} 年就开始了。', '{month} 那次家庭旅行时又提过。'],
  'chinese-work': ['这是 {year} 年定下的规矩。', '{month} 的季度复盘上又强调过一次。'],
  books: ['The rule was set in {year} and has held since.', 'It came up again during the {month} holiday.'],
}
const EN_TAILS = ['It came up again in {year}.', 'This was reaffirmed during the {month} review in {city}.']

function fill(template, rng) {
  return template
    .replaceAll('{year}', pick(rng, YEARS))
    .replaceAll('{years}', pick(rng, VARY_YEARS))
    .replaceAll('{city}', pick(rng, CITIES))
    .replaceAll('{name}', pick(rng, NAMES))
    .replaceAll('{amount}', pick(rng, AMOUNTS))
    .replaceAll('{month}', pick(rng, MONTHS))
    .replaceAll('{game}', pick(rng, GAMES))
    .replaceAll('{km}', pick(rng, KMS))
}

/**
 * 生成 N 条唯一记忆文本（确定性）。
 * 轮转族与模板、槽位由 RNG 填充，拼接族尾巴从句；Set 去重保证唯一（冲突重掷）。
 * 返回 { texts, familyKeys }（familyKeys[i] = texts[i] 所属族，供查询/评测对齐用）。
 */
export function generateCorpus(n, seed = 42) {
  const rng = mulberry32(seed)
  const texts = []
  const familyKeys = []
  const seen = new Set()
  let i = 0
  while (texts.length < n) {
    const family = FAMILIES[i % FAMILIES.length]
    const template = family.templates[Math.floor(rng() * family.templates.length)]
    const tail = TAILS[family.key] ?? EN_TAILS
    const text = fill(template, rng) + ' ' + fill(pick(rng, tail), rng)
    if (!seen.has(text)) {
      seen.add(text)
      texts.push(text)
      familyKeys.push(family.key)
    }
    i++
    if (i > n * 30) throw new Error('corpus generation stalled: too many duplicates')
  }
  return { texts, familyKeys }
}

/** 每族一条查询（措辞与库内模板不同但共享族词汇），返回 { queries, familyKeys }。 */
export function generateQueries(seed = 4242) {
  const queries = []
  const familyKeys = []
  for (const family of FAMILIES) {
    for (const q of QUERY_BY_FAMILY[family.key]) {
      queries.push(q)
      familyKeys.push(family.key)
    }
  }
  return { queries, familyKeys }
}

const QUERY_BY_FAMILY = {
  database: ['Which database does the team prefer for new services?', 'How are PostgreSQL schema migrations handled?'],
  editor: ['What editor and keymap does the user prefer?', 'How are Neovim plugins managed?'],
  coffee: ['How does the user brew coffee in the morning?', 'What is the rule about espresso after 3pm?'],
  deploy: ['How do deployments and rollbacks work for this team?', 'What changed in the deploy process after a failure?'],
  python: ['What are the user habits around Python typing and testing?', 'How is Python dependency management done?'],
  running: ['What is the user marathon training plan?', 'When do running shoes get replaced?'],
  finance: ['How does the user manage the household budget?', 'What is the emergency fund policy?'],
  gaming: ['What are the user rules about gaming and microtransactions?', 'What is the gaming setup?'],
  'chinese-food': ['用户的饮食偏好有哪些？', '用户对外卖备注有什么固定习惯？'],
  'chinese-family': ['用户的家庭安排有哪些固定节奏？', '用户家里最近在学什么乐器？'],
  'chinese-work': ['用户团队的工作节奏和惯例是什么？', '用户在推进什么基础设施迁移？'],
  books: ['What are the user reading habits and rules?', 'How does the user buy books?'],
}

export const FAMILY_KEYS = FAMILIES.map((f) => f.key)

// ---------------------------------------------------------------- 成簇语料（睡眠周期专用）

/**
 * 成簇语料：每组一条基句 + 近重复变体（换槽位值 / 组内尾巴），模拟
 * 「同一件事的多次近重复情景记忆」——睡眠周期压缩的真实输入形态。
 * 与 FAMILIES 通用语料相反，这里组内 token 重叠压到 ~80%，
 * deterministic 嵌入下组内 pairwise 余弦 ~0.6-0.9，能通过
 * clusterCos + 簇内均值闸门（阈值由 bench-sleep 现场校准）。
 */
const GROUP_BASES = [
  { key: 'commute', base: 'The user takes the {num7} bus from {city} to the office every weekday morning and always uses the same transit card.' },
  { key: 'backup', base: 'The user backs up the {sys} database to {dest} every night and checks the backup log the next morning.' },
  { key: 'standup', base: 'The user runs the team standup at {num2} in the {room} room and always starts with the same three questions.' },
  { key: 'gym', base: 'The user does a {min} minute {lift} session at the gym near {city} on the same three weekdays.' },
  { key: 'tea', base: '用户每天下午用同一只{cup}泡{tea}，泡完会把茶叶罐放回办公室左手边第二个抽屉。' },
  { key: 'report', base: 'The user files the weekly {sys} status report before {num2} on Friday and cc-es the same two reviewers.' },
  { key: 'groceries', base: 'The user buys {fruit} and {fruit2} at the market near {city} every Sunday morning before ten.' },
  { key: 'lint', base: 'The user runs the {sys} linter gate before every merge and never approves a diff with new warnings.' },
  { key: 'journal', base: '用户每晚睡前在同一个本子上写三行日记，写完把本子放回床头柜第一层。' },
  { key: 'oncall', base: 'The user takes the on-call rotation for the {sys} service every third week and hands over with the same checklist.' },
  { key: 'backup2zh', base: '用户每周六上午把{sys}的笔记导出备份到同一块移动硬盘，备份完会看一眼剩余容量。' },
  { key: 'coffeebreak', base: 'The user takes a {min} minute coffee break at {num2} every afternoon and always brings the same mug from home.' },
]

const GROUP_TAILS = {
  commute: ['This routine has been stable for years.', 'The bus is usually on time.'],
  backup: ['The backup has never failed so far.', 'Restores are tested monthly.'],
  standup: ['The team knows the format by heart.', 'Latecomers get a recap after.'],
  gym: ['The routine skips only on vacation.', 'Progress is logged in an app.'],
  tea: '这个习惯已经保持很久了。'.split('|'),
  report: ['The template lives in the team wiki.', 'Reminders fire two hours before.'],
  groceries: ['The market opens early on Sundays.', 'Cash is preferred at this market.'],
  lint: ['The gate runs in CI automatically.', 'Exceptions need a written waiver.'],
  journal: '偶尔漏写也不会补。'.split('|'),
  oncall: ['The handover doc is versioned.', 'Pages are rare on this service.'],
  backup2zh: '硬盘放在书桌右手边。'.split('|'),
  coffeebreak: ['The break is never skipped.', 'Colleagues join most days.'],
}

const NUMS7 = ['27', '31', '58', '66']
const NUMS2 = ['09:30', '10:00', '10:15', '14:00']
const MINS = ['30', '45', '60']
const SYSES = ['orders', 'billing', 'search', 'auth']
const DESTS = ['the nas box', 'the cold storage bucket', 'an external drive']
const ROOMS = ['glass meeting', 'small corner', 'north']
const LIFTS = ['barbell', 'kettlebell', 'bodyweight']
const CUPS = ['白瓷杯', '玻璃杯']
const TEAS = ['茉莉花茶', '龙井']
const FRUITS = ['apples', 'pears', 'oranges']
const ZH_SYS = ['工作笔记', '个人笔记']

function fillGroup(base, tail, rng) {
  return (base + ' ' + tail)
    .replaceAll('{num7}', pick(rng, NUMS7))
    .replaceAll('{num2}', pick(rng, NUMS2))
    .replaceAll('{min}', pick(rng, MINS))
    .replaceAll('{sys}', pick(rng, SYSES))
    .replaceAll('{syszh}', pick(rng, ZH_SYS))
    .replaceAll('{dest}', pick(rng, DESTS))
    .replaceAll('{room}', pick(rng, ROOMS))
    .replaceAll('{lift}', pick(rng, LIFTS))
    .replaceAll('{city}', pick(rng, CITIES))
    .replaceAll('{cup}', pick(rng, CUPS))
    .replaceAll('{tea}', pick(rng, TEAS))
    .replaceAll('{fruit2}', pick(rng, FRUITS))
    .replaceAll('{fruit}', pick(rng, FRUITS.filter((f) => f !== 'apples')))
}

/**
 * 生成 n 条成簇文本（确定性）：groupIdx[i] 标注所属组（0-based）。
 * 唯一性兜底：冲突时追加组无关的序号词（唯一 token，对组内余弦影响 <5%）。
 */
export function generateClusteredCorpus(n, seed = 555) {
  const rng = mulberry32(seed)
  const texts = []
  const groupIdx = []
  const seen = new Set()
  let i = 0
  while (texts.length < n) {
    const g = i % GROUP_BASES.length
    const group = GROUP_BASES[g]
    const tails = GROUP_TAILS[group.key]
    let text = fillGroup(group.base, tails[Math.floor(rng() * tails.length)], rng)
    if (seen.has(text)) text += ` Variant number ${texts.length} of this recurring routine.`
    if (!seen.has(text)) {
      seen.add(text)
      texts.push(text)
      groupIdx.push(g)
    }
    i++
    if (i > n * 30) throw new Error('clustered corpus generation stalled')
  }
  return { texts, groupIdx, groupCount: GROUP_BASES.length }
}

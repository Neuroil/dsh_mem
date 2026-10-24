// 只读检查 sqlite 记忆库内容
import { DatabaseSync } from 'node:sqlite'

const path = 'C:/Users/Neuroil/.dsh/storages/dsh_mem.sqlite'
const db = new DatabaseSync(path, { readOnly: true })
const tabs = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all()
console.log('tables:', tabs.map((t) => t.name).join(','))
for (const t of tabs) {
  try {
    const c = db.prepare(`SELECT COUNT(*) AS n FROM "${t.name}"`).get()
    console.log(' ', t.name, 'rows=' + c.n)
  } catch (e) {
    console.log(' ', t.name, 'ERR', e.message)
  }
}
db.close()

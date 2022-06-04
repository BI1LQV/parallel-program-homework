enum REQ_MSG_TYPE {
  NO_SIGNAL = 0,
  CONNECT = -1,
  REQUEST_TASK = -2,
  REPORT_RES = -3,
};

enum MSG_SYMBOL {
  MSG_END = -4,
  END_CALC = -5
};

interface ConnWithId extends Deno.Conn {
  id?: number
}

const BATCH_SIZE = 60000

const results = new Map<number, number[]>()
let startTime: number = -1
const MAX_DEVICE = 1
let endCalcCount = 0
let deviceLock = (() => {
  let deviceNum = 0
  let resolvePool = new Array<(value: unknown) => void>()
  return () => {
    if (startTime !== -1) {
      return true
    }
    const lock = new Promise((resolve, reject) => {
      resolvePool.push(resolve)
    })
    if (++deviceNum >= MAX_DEVICE) {
      //发布任务 开始计算
      resolvePool.forEach(resolve => resolve(true))
      startTime = performance.now()
      console.log('-------start calc-------', startTime)
    }
    return lock
  }
})()
// const connectionPool = new Map<number, Deno.Conn>()
const getId = (() => {
  let id = 0
  return () => id++
})()
const getTask = (() => {
  let tasks = new Set(Array.from({ length: 60000 / BATCH_SIZE }, (_, i) => i))
  let iter = tasks[Symbol.iterator]()
  return () => iter.next()
})()
const eventHandler = {
  CONNECT: async (req: Int32Array, res: ConnWithId) => {
    const response = [getId()]
    // connectionPool.set(response[0], res)
    res.id = response[0]
    await writeConn(res, response)
  },
  REQUEST_TASK: async (req: Int32Array, res: ConnWithId) => {
    let nextTask = getTask()
    let response: number[]
    if (nextTask.done) {//计算完成 结束
      response = [MSG_SYMBOL.END_CALC]
      if (++endCalcCount >= MAX_DEVICE) {
        console.log('-------end calc-------', performance.now() - startTime)
      }
    } else {
      response = [nextTask.value]
    }
    await deviceLock()
    await writeConn(res, response)
  },
  REPORT_RES: async (req: Int32Array, res: ConnWithId) => {
    let arrayedRes = Array.from(req)
    let taskId = arrayedRes[1]
    results.set(taskId, arrayedRes.slice(2, arrayedRes.indexOf(MSG_SYMBOL.MSG_END)))
    console.log('receiving', taskId);
    if (results.size === 60000 / BATCH_SIZE) {
      console.log('-------test-------')
      let sumCol = [];
      Array.from({ length: 60000 / BATCH_SIZE }, (_, i) => i).forEach((i: number) => {
        sumCol.push(...results.get(i)!)
      })
      console.log(sumCol.length)
      Deno.exit(0)
    }
    // for (let conn of connectionPool.values()) {
    //   writeConn(conn, [MSG_SYMBOL.END_CALC])
    // }
  }
}

const listener = Deno.listen({ port: 1234 })

console.log("listening on 0.0.0.0:1234")


const writeConn = async (conn: Deno.Conn, data: number[]) => {
  const resBuffer = new ArrayBuffer(data.length * 4)
  new Int32Array(resBuffer).set(data)
  console.log(`respond cid:${'id' in conn ? (conn as ConnWithId).id : '_'} time:`, Date.now(), 'data:', new Int32Array(resBuffer))
  await conn.write(new Uint8Array(resBuffer))
}

for await (const conn of listener) {
  const cycle = async () => {
    const req = new ArrayBuffer(BATCH_SIZE * 4 + 32)
    await conn.read(new Uint8Array(req))
    const reqViewer = new Int32Array(req)
    const eventType = (REQ_MSG_TYPE[reqViewer[0]] as keyof typeof REQ_MSG_TYPE)
    if (eventType === "NO_SIGNAL") {
      return
    }
    console.log(`request cid:${'id' in conn ? (conn as ConnWithId).id : '_'} time:`, Date.now(), 'event:', eventType, reqViewer[0])
    await eventHandler[eventType]?.(reqViewer, conn)
    cycle()
  }
  cycle()
}
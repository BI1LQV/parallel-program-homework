import net from 'node:net'
let MSG_TYPE =
    [
	/*CONNECT = */'0',
	/*REQUEST_TASK = */  '1',
	/*REPORT_RES = */  '2'
    ]
let incId = 0
const server = new net.Server({ keepAlive: true }, (socket) => {
    socket.pipe(socket)
    socket.on('data', (data) => {
        let type = MSG_TYPE[data[0]]
        switch (type) {
            case '0':
                console.log('CONNECT')
                socket.write('CONNECT')
                break;
        }
    })
    // console.log(socket)
});
server.listen(1234)
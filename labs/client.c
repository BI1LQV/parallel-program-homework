#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

enum REQ_MSG_TYPE
{
    CONNECT = -1,
    REQUEST_TASK = -2,
    REPORT_RES = -3,
};

enum MSG_SYMBOL
{
    MSG_END = -4,
    END_CALC = -5,
};

int main()
{
    //创建套接字
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serv_addr.sin_port = htons(1234);
    connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
    int clientId;
    int msgs[] = {CONNECT};

    send(sock, msgs, sizeof(msgs), 0);
    read(sock, &clientId, sizeof(clientId));
    printf("id %d\n", clientId);
    int *taskList = (int *)malloc(1000 * 4);
    int taskListPtr = 0;
    for (int i = 0; i < 10; i++)
    {
        sleep(5);
        msgs[0] = REQUEST_TASK;
        send(sock, msgs, sizeof(msgs), 0);
        int taskId;
        read(sock, &taskId, sizeof(taskId));
        taskList[taskListPtr++] = taskId;
        if (taskId == END_CALC)
        {
            printf("endcalc\n");
            for (int t = 0; t < taskListPtr; t++)
            {
                if (taskList[t] == END_CALC)
                {
                    break;
                }
                printf("%d %d\n", t, taskList[t]);
                int res[] = {-3, taskList[t], 2, 2, 3, 1, 1, 2, 3};
                int sent = send(sock, res, sizeof(res), 0);
                sleep(1);
                printf("sent %d\n", sent);
            }
            printf("sent\n");
            return 0;
        }
        printf("task id %d\n", taskId);
    }
    // while (1)
    // {
    //     read(sock, buffer, sizeof(buffer) - 1);
    //     // char hello[] = "hello!";
    //     // send(sock, hello, strlen(hello), 0);
    //     printf("Message form server: %s\n", buffer);
    // }
    //关闭套接字
    close(sock);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>

#include <omp.h>

#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void minuim(int *requestTask)
{
    sleep(3);
    printf("min in\n");
    *requestTask = 1;
    printf("changed\n");
}
void comm(int *requestTask, int sock, int *taskId)
{
    int req[] = {-2};
    while (1)
    {
        if (*requestTask == 1)
        {
            printf("reasdfdsafsdafsdafsdafsadfsda\n");
            send(sock, req, sizeof(req), 0);
            printf("reasdfsda\n");
            read(sock, taskId, sizeof(*taskId));
            printf("red%d\n", *taskId);
            *requestTask = 0;
            printf("requestd %d\n", *taskId);
        }
    }
}

int main()
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("192.168.1.253");
    serv_addr.sin_port = htons(1234);
    connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    int clientId;
    int msgs[] = {-1};

    send(sock, msgs, sizeof(msgs), 0);
    read(sock, &clientId, sizeof(clientId));
    printf("this clientid %d\n", clientId);

    int requestTask = 1;
    int taskId = -1;
#pragma omp parallel num_threads(2) shared(requestTask)
    {
#pragma omp single
        {
#pragma omp task
            comm(&requestTask, sock, &taskId);
#pragma omp task
            minuim(&requestTask);
        }
    }
    return 0;
}
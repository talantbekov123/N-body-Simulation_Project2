#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
typedef struct _body {
        float x, y;
        float ax, ay;
        float vx, vy;
        float mass;
} body;
const int N = 20;
body *bodies;

void calculate_newton_gravity_acceleration(body *firstBody, body *secondBody, float *ax, float *ay) {
        float vectorOfDistance = sqrt(firstBody->x * firstBody->x + firstBody->y * firstBody->y) + 10;
        float distanceSquaredCubed = vectorOfDistance * vectorOfDistance * vectorOfDistance;
        float scale = secondBody->mass * (1.0 / distanceSquaredCubed);
        *ax = (secondBody->x - firstBody->x) * scale;
        *ay = (secondBody->y - firstBody->y) * scale;
}
void integrate(body *body, float delta_Time) {
        body->x += body->vx * delta_Time;
        body->y += body->vy * delta_Time;
        body->vx += body->ax * delta_Time;
        body->vy += body->ay * delta_Time;
}

int main(int argc, char **argv) {
        float delta_Time = 0.1;
        bodies = (body*) malloc(N * sizeof(*bodies));
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            float angle = ((float) i / N) * 2.0 * 3.14159265358979323846 + (rand() % 2 * 0.1 - 0.5) * 0.5;
            float initialMass = 3.0;
            body body0;
            body0.x = rand() % 2 * 0.1;
            body0.y = rand() % 2 * 0.1;
            body0.vx = cos(angle) * 100 * (rand() % 2 * 0.1);
            body0.vy = sin(angle) * 100 * (rand() % 2 * 0.1);
            body0.mass = initialMass * (rand() % 2 * 0.1) + initialMass * 0.5;
            float scale = body0.mass / (initialMass * 1.5) + 0.1;
            bodies[i] = body0;
        }
        
		MPI_Init(&argc, &argv);
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int bodies_per_process = N / world_size;
        if(rank == 0) {
            printf("%d\n", N);
            printf("%f\n", delta_Time);
            for(int j = 0; j < N; j++) {
                printf("%f %f\n", bodies[j].x,  bodies[j].y);
                printf("%f %f\n", bodies[j].ax,  bodies[j].ay);
                printf("%f %f\n", bodies[j].vx,  bodies[j].vy);
                printf("%f\n", bodies[j].mass);
            }
        }
        MPI_Datatype body_datatype;
        MPI_Type_contiguous(7, MPI_FLOAT, &body_datatype);
        MPI_Type_commit(&body_datatype);
        body *local_bodies = (body *) malloc(sizeof(*local_bodies) * bodies_per_process);
        MPI_Scatter(bodies, bodies_per_process, body_datatype, local_bodies, bodies_per_process, body_datatype, 0, MPI_COMM_WORLD);

        for(double time = 0; time < 1; time += delta_Time) {
            for(int i = 0; i < bodies_per_process; i++) {
                float total_ax = 0.0, total_ay = 0.0;
                //printf("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF   %d\n", N);
                for (size_t j = 0; j < N; j++) {
                    if (i + rank * bodies_per_process == j) {continue;}
                    float ax = 0, ay = 0;
                    calculate_newton_gravity_acceleration(&local_bodies[i], &bodies[j], &ax, &ay);
                    total_ax += ax;
                    total_ay += ay;
                }
                local_bodies[i].ax = total_ax;
                local_bodies[i].ay = total_ay;
                printf("%f %f\n", local_bodies[i].ax, local_bodies[i].ay);
                integrate(&local_bodies[i], delta_Time);
            }
        }

        body *newBodies = (body*) malloc(N * sizeof(*newBodies));
        MPI_Gather(local_bodies, bodies_per_process, body_datatype, newBodies, bodies_per_process, body_datatype, 0, MPI_COMM_WORLD);
		if(local_bodies != NULL) {
            free(local_bodies);
        }
        if(bodies != NULL) {
            free(bodies);
        }
        if(newBodies != NULL) {
            free(newBodies);
        }
        MPI_Finalize();
        return 0;
}

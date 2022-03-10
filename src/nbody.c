#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "mpi.h"

#define SOFTENING 1e-9f
typedef struct { float x, y, z, vx, vy, vz; } Body;
MPI_Datatype NewBodies;
Body *particella,*porzione;
 
int modulo;
int numElementi;
 
int rank;
int size;
MPI_Status status;


int ControllaInput(int argc, char *argv[]);
void InizializzaBodies();
void bodyForce(Body *porzione, int n);
void Stampa_Risultati();

int nCorpi;
int nIterazioni; 
const float dt = 0.01f;
int *counts,*displacements;

int main(int argc, char *argv[])
{
  double start,end;
  //Start Up MPI
  MPI_Init (&argc, &argv);
  //Return the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //return the number of process
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Type_contiguous(6,MPI_FLOAT,&NewBodies); //creiamo un nuovo tipo di dato contiguos che useremo per le comunicazioni MPI
  MPI_Type_commit(&NewBodies); //Rendiamo effettiva l'allocazione di memoria
  MPI_Barrier(MPI_COMM_WORLD); //Aspetto che tutti i processi abbiamo eseguito le istruzioni fino a qui.
  int input=ControllaInput(argc,argv);
  if(input==0)
  {
    if(rank==0)
    {
      printf("\nValori di Input Mancanti\n");
      printf("Inserisci due valori:\n");
      printf("1. Numero di Particelle\n");
      printf("2. numero di cicli\n\n");
      printf("Ad esempio: 30000 10\n\n");
    }
    MPI_Finalize();
    return 0;
  }
  int bytes = nCorpi*sizeof(Body);  //Alloco la memoria che conterrà le particelle e le varie porzioni che saranno usate dai processi per l'elaborazione.
  float *buf_particella =malloc(bytes);
  float *buf_porzione =malloc(bytes);
  particella = (Body*)buf_particella;
  porzione = (Body*)buf_porzione;
  counts=(int*)malloc(sizeof(int)*size);
  displacements=(int*)malloc(sizeof(int)*size);
  start=MPI_Wtime();
  InizializzaBodies();  //Inizializzazione  
    
  modulo=nCorpi % size;
  numElementi=nCorpi/size;
  if(modulo==0)
  {
    MPI_Bcast(particella,nCorpi,NewBodies,0,MPI_COMM_WORLD);
    MPI_Scatter(particella,numElementi,NewBodies,porzione,numElementi,NewBodies,0,MPI_COMM_WORLD);
    bodyForce(porzione,numElementi);
  }
  else
  {
    /* Inizio Calcoli Necessari per usare MPI_Allgatherv */
    MPI_Bcast(particella,nCorpi,NewBodies,0,MPI_COMM_WORLD);
    int fine=modulo;
    int partenza=0;
    int elementi=numElementi+1;
    for(int k=0;k<size;k++)
    {
      if(fine>0)
      {
        counts[k]=elementi;
        displacements[k]=partenza;
        fine--;
        partenza=partenza+elementi;
      }
      else
      {
        counts[k]=numElementi;
        displacements[k]=partenza;
        partenza=partenza+numElementi;
      }
    }
    /* Fine Calcoli per MPI_Allgatherv */
     
    if(rank==0)  //Invio
    {
      int rimanenza=(modulo-1);
      int inizio=numElementi+1,lunghezza=0;
      for (int i=0;i<size-1;i++) //da provare
      {
        if(rimanenza>0)
        {
          lunghezza=numElementi+1;
          MPI_Send(&inizio,1,MPI_INT,i+1,99,MPI_COMM_WORLD);
          MPI_Send(&lunghezza,1,MPI_INT,i+1,99,MPI_COMM_WORLD);
          inizio=inizio+lunghezza;
          rimanenza--;
        }
        else
        {
          MPI_Send(&inizio,1,MPI_INT,i+1,99,MPI_COMM_WORLD);
          MPI_Send(&numElementi,1,MPI_INT,i+1,99,MPI_COMM_WORLD);
          inizio=inizio+numElementi;
        }
      }

      //Computazione Rank 0
      inizio=0;
      for(int j=0;j<numElementi+1;j++)
      {
        porzione[j]=particella[inizio];
        inizio++;
      }
      bodyForce(porzione,numElementi+1);
      //Fine Computazione Rank 0
    }
    else // Ricezione
    {
      int inizio=numElementi+1,lunghezza=0;
      MPI_Recv(&inizio,1,MPI_INT, 0,99, MPI_COMM_WORLD, &status);
      MPI_Recv(&lunghezza,1,MPI_INT, 0,99, MPI_COMM_WORLD, &status);
      for(int j=0;j<lunghezza;j++)
      {
        porzione[j]=particella[inizio];
        inizio++;
      }
      bodyForce(porzione,lunghezza);
      printf("\n");
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0)
  {
    end=MPI_Wtime();
    Stampa_Risultati();
    printf("\t--> Complete in: %f <-- \n\n",end-start);
  }
  MPI_Type_free(&NewBodies);
  free(particella);
  free(porzione);
  MPI_Finalize();
  return 0;
}

int ControllaInput(int argc, char *argv[])
{
  if (argc > 2)
  {
    nCorpi = atoi(argv[1]);
    nIterazioni = atoi(argv[2]);
  } 
  else
  {
    return 0;
  }   
}

void InizializzaBodies() 
{
  srand(10); //usato per il testing per avere sempre gli stessi valori nell'inizializzazione
  for (int i = 0; i < nCorpi; i++) 
  {
    particella[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    particella[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    particella[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    particella[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    particella[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    particella[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *porzione, int n) 
{
  for(int it=0;it<nIterazioni;it++)
  {
    for (int i = 0; i < n; i++) 
    { 
      float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
      for (int j = 0; j<nCorpi; j++) 
      {
        float dx = particella[j].x - porzione[i].x;
        float dy = particella[j].y - porzione[i].y;
        float dz = particella[j].z - porzione[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; 
        Fy += dy * invDist3; 
        Fz += dz * invDist3;
      }
      porzione[i].vx += dt*Fx; 
      porzione[i].vy += dt*Fy; 
      porzione[i].vz += dt*Fz;
    }
    for (int i=0 ; i < n; i++) 
    {
      porzione[i].x += porzione[i].vx*dt;
      porzione[i].y += porzione[i].vy*dt;
      porzione[i].z += porzione[i].vz*dt;  
    }
    if(modulo==0)
      MPI_Allgather(porzione,n,NewBodies,particella,n,NewBodies,MPI_COMM_WORLD);
    else
      MPI_Allgatherv(porzione,n,NewBodies,particella,counts,displacements,NewBodies,MPI_COMM_WORLD);
  }  
}

void Stampa_Risultati()
{
  for(int i=0;i<nCorpi;i++)
  {
    printf("\n");
    printf("\t[ # %d ] [ x: %f ] [ y: %f ] [ z: %f ] [ vx: %f ] [ vy: %f ] [ vz: %f ]\t\n",i+1,particella[i].x,particella[i].y,particella[i].z, particella[i].vx, particella[i].vy, particella[i].vz);
    printf("\n\n");
  }
}

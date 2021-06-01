# N-Body Simulation
***
## Programmazione Concorrente, Parallela e su Cloud
### Università degli studi di Salerno
#### *Anno Accademico 2020/2021*
**Professore:** *Vittorio Scarano,* **Professore:** *Carmine Spagnuolo*
**Studente:** *Galante Giuseppe*
**Matricola:** *0522500917*
***
## Problem Statement
Nel N-body problem, dobbiamo trovare le posizioni e le velocità di un insieme di particelle che interagiscono nel tempo. Ad esempio, un astrofisico potrebbe voler conoscere le posizioni e le velocità di un gruppo di stelle. Al contrario, un chimico voler conoscere le posizioni e le velocità di un insieme di molecole o atomi.
## Soluzione Proposta
Al N-body solver vengono dati in input il numero di particelle e le iterazioni da simulare. Per quanto concerne i dettagli relativi alle particelle (posizioni x,y,z iniziali nello spazio e velocità di propagazione) sono generate in modo casuale. Al termine del numero di iterazioni in output restituirà la posizione (x,y,z) e la velocità per ogni particella. La soluzione proposta prende in considerazione solo l' approccio quadratico sul numero di particelle.
Il software si avvale di comunicazione sia collettiva (**MPI_Bcast**, **MPI_Scatter**, **MPI_Allgather** e **MPI_Allgatherv**) sia Point-to-Point (**MPI_Send** e **MPI_Recv**).
I test sono stati effettuati su istanza di AWS **m4.xlarge**.
## Implementazione
L'obiettivo del progetto è quello di parallelizzare nel miglior modo possibile l'algoritmo N-body sequenziale. Per ottenere il miglior risultato andremo a dividere il numero di particelle in modo equo fra tutti i processi coinvolti. Per raggiungere questo obiettivo abbiamo applicato il metodo seguente.
#### Variabili
**Body:** contiene tutte le informazioni di una particella.
```
typedef struct { float x, y, z, vx, vy, vz; } Body;
```
**Variabili utili per il funzionamento di MPI:**
```
int rank; //rank of process
int size; //number of process
MPI_Status status; //return status for receive
```
**Parametri passati da riga di comando:**
```
int nCorpi;   //numero di particelle
int nIterazioni; //numero di iterazioni
```
#### ControllaInput
Controlla se il numero di parametri è corretto.
```
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
```
#### InizializzaBodies
Inizializza tutte le caratteristiche delle particelle.
```
void InizializzaBodies() 
{
  srand(10); //Used to have the same input for the testing phase
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
```
#### bodyForce
Calcola gli spostamenti delle particelle per una porzione di bodies passati al processo.
- porzione: l'insieme delle particelle che un processo deve computare;
- n: numero di particelle che _porzione_ contiene.

Per ogni particella contenuta in _porzione_, si vanno a calcolare le posizioni (x,y,z) e le velocità (vx,vy,vz) dopo la simulazione di _it_ iterazioni.
Al termine della computazione si usa la funzione collettiva **_MPI_Allgather_** nel caso in cui il numero di particelle sia divisibile perfettamente per il numero di processi usati o la funzione **_MPI_Allgatherv_**  nel caso non lo sia. L'uso delle suddette funzioni ha lo scopo di propagare agli altri processi i nuovi valori calcolati.
```
void bodyForce(Body *porzione,int n) 
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
```
#### Stampa_Risultati
Stampa in modo gradevole per l'utente i risultati ottenuti.
```
void Stampa_Risultati()
{
  for(int i=0;i<nCorpi;i++)
  {
    printf("\n");
    printf("\t[ # %d ] [ x: %f ] [ y: %f ] [ z: %f ] [ vx: %f ] [ vy: %f ] [ vz: %f ]\t\n",i+1,particella[i].x,particella[i].y,particella[i].z, particella[i].vx, particella[i].vy, particella[i].vz);
    printf("\n\n");
  }
}
```
### Il main:
Le prime operazioni effettuate del main sono quelle di inizializzazione di MPI.
```
//Start Up MPI
MPI_Init (&argc, &argv);
//Return the rank of the process
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//return the number of process
MPI_Comm_size(MPI_COMM_WORLD, &size);
```
Fatto ciò con la funzione ***_MPI_Type_contiguous_*** creiamo un nuovo tipo di dato contiguos _NewBodies_ che useremo per le comunicazioni MPI. Successivamente con ***_MPI_Type_commit_*** rendiamo effettiva l'allocazione di memoria.
```
MPI_Type_contiguous(6,MPI_FLOAT,&NewBodies);
MPI_Type_commit(&NewBodies);
```
La seguente parte effettua i controlli sull'input e nel caso non siano rispettati termina il programma avvisando l'utente.
```
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
```
Alloco la memoria che conterrà le particelle e le varie porzioni che saranno usate dai processi per l'elaborazione.
```
int bytes = nCorpi*sizeof(Body);
float *buf_particella =malloc(bytes);
float *buf_porzione =malloc(bytes);
particella = (Body*)buf_particella;
porzione = (Body*)buf_porzione;
```
Alloco la memoria che conterrà il numero di elementi che devono essere ricevuti da ogni processo ```counts``` e la memoria che conterrà la posizione da cui iniziare l'inserimento degli elementi che sono ricevuti da ogni processo ```displacements``` per il funzionamento di **_MPI_Allgatherv_**.
```
counts=(int*)malloc(sizeof(int)*size);
displacements=(int*)malloc(sizeof(int)*size);
```
Inizializziamo il tempo di esecuzione iniziale
```
start=MPI_Wtime();
```
E le particelle
```
InizializzaBodies();
```
Calcoliamo se il numero di particelle è divisibile perfettamente per il numero di processi o meno e la quantità di particelle che ogni processo deve ricevere. 
```
modulo=nCorpi % size;
numElementi=nCorpi/size;
```
Se il modulo è pari a 0 allora tutti i processi devono ricevere la stessa quantità di particelle e quindi possiamo usare le funzioni collettive **Bcast** e **Scatter** per l'invio. La prima invia tutte le particelle a tutti i processi e la seconda divide in parti uguali tra i processi le particelle da computare. Successivamente viene richiamato il metodo **bodyForce**.
```
MPI_Bcast(particella,nCorpi,NewBodies,0,MPI_COMM_WORLD);
MPI_Scatter(particella,numElementi,NewBodies,porzione,numElementi,NewBodies,0,MPI_COMM_WORLD);
bodyForce(porzione,numElementi);
```
Se il modulo è diverso da 0, per prima cosa usiamo la funzione collettiva **Bcast** per inviare a parità del caso precedente tutte le particella a tutti i processi e successivamente popolo ```counts``` e ```displacements``` in modo da poter utilizzare correttamente **MPI_Allgatherv** in ```bodyForce```. 
```
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
```
Successivamente si preparano le varie porzioni di particelle che saranno successivamente inviate ai processi da parte del processo *master 0*.
#### Invio ####
Per prima cosa si verifica che ci siano delle particelle da aggiungere quando la divisone particelle/processi genera un resto; in questo caso si calcolano lunghezza e punto di inizio delle particelle che tengono conto delle aggiunte necessarie, altrimenti si inviano semplicemente senza modifiche. Dopo aver effettuato i dovuti calcoli, in un caso o nell'altro, si usa la funzione Point-to-Point **MPI_Send** per inviare le grandezze a tutti gli altri processi.
```
int rimanenza=(modulo-1);
int inizio=numElementi+1,lunghezza=0;
for (int i=0;i<size-1;i++)
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
```
Terminato l'invio delle grandezze, si forma la porzione che conterrà le particelle che lo stesso master andrà a computare richiamando **_bodyForce_**.
```
for(int j=0;j<numElementi+1;j++)
{
    porzione[j]=particella[inizio];
    inizio++;
}
bodyForce(porzione,numElementi+1);
```

#### Ricezione ####
La ricezione comincia usando la funzione Point-to-Point **MPI_Recv** per recuperare le dimensioni inviate attraverso la *Send*. Successivamente si crea la porzione da computare (una per ogni processo) andando infine a computarla effettivamente con la chiamata della funzione **_bodyForce_**.
```
for(int i=1;i<size;i++)
{
    MPI_Recv(&inizio,1,MPI_INT, 0,99, MPI_COMM_WORLD, &status);
    MPI_Recv(&lunghezza,1,MPI_INT, 0,99, MPI_COMM_WORLD, &status);
    for(int j=0;j<lunghezza;j++)
    {
        porzione[j]=particella[inizio];
        inizio++;
    }
    bodyForce(porzione,lunghezza);
    printf("\n");
    break;
}
```
Terminata tutta la parte di divisione ed invio delle particelle ci rimane da aspettare la terminazione dei processi tramite la **MPI_Barrier**, calcolare il tempo di esecuzione recuperando il tempo di terminazione con ```end=MPI_Wtime();```, richiamare la funzione ``` Stampa_Risultati(); ``` per stampare le particelle con le proprietà aggiornate,  liberare la memoria dal tipo allocato ``` NewBodies ```, da ```particella ``` e ``` porzione ```. Ed infine terminare MPI e con esso l'intera simulazione con ```  MPI_Finalize(); ```

```
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
```
## Testing
I test sono stati effettuati sulle istanze **m4.xlarge** (4 core) di Amazon Web Service (AWS). Durante tutta la fase di testing si è tenuto conto sia dello strong scaling che del weak scaling.

Risorse Utilizzate:
 - 8 istanze EC2 m4.xlarge **Ubuntu Server 20.04 LTS (HVM)** ami-09e67e426f25ce0d7 (64bit (x86))
 - 32 processori (4 core per istanza)

I test sono stati effettuati con i seguenti parametri:
- Numero Iterazioni pari a 30
- Istante di tempo pari a 0.01

``` 
const float dt = 0.01f; 
```
Per il calcolo dello strong scaling e del weak scaling sono state applicate le formule descritte al link seguente: [Measuring Parallel Scaling Performance](https://www.sharcnet.ca/help/index.php/Measuring_Parallel_Scaling_Performance "Strong e Weak Scaling").

### Strong Scalability
Nella fase di testing che ha tenuto conto dello strong scaling sono state utilizzate 60000 particelle e 30 iterazioni. Nello strong scaling infatti il numero di particelle è fisso, quello che varia è il numero di processori (nel nostro caso da 1 a 32).
Nel grafico seguente è possibile osservare i risultati ottenuti.

![Strong Scaling](https://github.com/GiuseppeGalante/NBody_MPI/blob/main/img/strong.png "Strong Scaling")

### Weak Scalability
La fase di testing che ha tenuto conto del weak scaling ha usato per il primo esperimento 5000 particelle e 30 iterazioni per processo (nel nostro caso da 1 a 32 processi) e nel secondo 10000 particelle e 30 iterazioni (1-14 poi 16 e 32 processi). Nel weak scaling infatti il numero di particelle cresce proporzionalmente al numero di processori.
Nel grafico seguente è possibile osservare i risultati ottenuti.

![Weak Scaling](https://github.com/GiuseppeGalante/NBody_MPI/blob/main/img/weak.png "Weak Scaling")

## Compilazione ed esecuzione del codice
Il codice va compilato con l'istruzione seguente

``` mpicc nbody.c -lm -o nbody ```

Per quanto riguarda invece l'esecuzione, l'istruzione da lanciare è la seguente.

``` mpirun -np <numero processori> --hostfile hfile ./nbody <numero particelle> <numero iterazioni> ```

A volte durante l'esecuzione su Docker potrebbero comparire degli errori simili ai seguenti:
```
Read -1, expected <someNumber>, errno =1
Read -1, expected <someNumber>, errno =1
Read -1, expected <someNumber>, errno =1
Read -1, expected <someNumber>, errno =1
Read -1, expected <someNumber>, errno =1
......
```
In tal caso bisogna aggiungere il parametro ```--mca btl_vader_single_copy_mechanism none```.
## Conclusioni
Come visto dai grafici sopra riportati, sia nella strong scalability che nella weak scalability, il grafico risultante dai tempi di esecuzione appare omogeneo; questo è dovuto al fatto che buona parte della soluzione è formata da metodi già messi a disposizione da MPI, le aggiunte permettono solo un giusto funzionamento delle stesse.


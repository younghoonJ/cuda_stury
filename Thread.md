# CUDA 스레드 계층
    스레드  < 워프 < 블록 < 그리드


### 워프
- 32개의 스레드를 하나로 묶은 단위. 
- cuda의 기본 수행 단위.
- 하나의 제어 장치로 제어됨.


### 블록
- 하나의 블록에 속한 스레드는 고요한 스레드 번호를 가짐
- 각 블록은 고유한 번호(block ID)를 가짐.
- 원하는 스레드를 지칭하려면 블록 번호와 스레드 번호 모두 필요.
- 1, 2, 3차원 형태로 배치 가능.


### 그리드
- 여러 개의 블록을 포함하는 블록들의 그룹.
- 1, 2, 3차원 형태로 배치 가능.
- 하나의 그리드에 속한 블록들은 고유한 block ID를 가짐.
- 커널 호출시 그리드 생성됨. 하나의 그리드는 하나의 커널 호출과 1:1 대응


## 내장 변수
    스레드들은 자신이 처리할 데이터가 무엇인지 알아야 함.
- gridDim: 그리드의 형태 정보.
- blockIdx: 현재 스레드가 속한 블록의 번호.
- blockDim: 블록의 형태 정보.
- threadIdx: 블록 내에서 현재 스레드가 부여받은 스레드 번호.


## 스레드 번호와 워프의 구성
- 워프는 연속된 32개 스레드로 구성됨
- 스레드 연속성은 threadIdx의 x,y,z 차원 순으로 결정됨. (0,0,0)~(31,0,0)번 스레드가 하나의 워프를 구성. 
- 만약 x차원의 길이가 워프의 크기보다 작다면 y차원의 번호가 낮은 순으로 연속성을 가짐. 만약 x차원이 1이면 (0,0,0)~(0,31,0) 순서.


## 그리드의 크기
- 최대 3차원. 
- x: 2^31 - 1
- y,z: 65535


## 블록의 크기
- x,y: 1024
- z: 64
- 블록 하나는 최대 1024개의 스레드까지 가질 수 있음.


## 스레드 레이아웃 설정 및 커널 호출
- Kernel<<<그리드 형태, 블록의 형태>>>()
- Dim3 구조체 사용.
```c
Dim3 dimGrid(4, 1, 1);
Dim3 dimBlock(8, 1, 1);
kernel<<<dimGrid, dimBlock>>>();
```


## 블록 내 스레드 전역 번호
- 1차원 블록: threadIdx.x == 스레드 전역 번호
- 2차원 블록: blockDim.x * threadIdx.y + threadIdx.x
- 3차원 블록: blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x

## 그리드 내 스레드의 전역 번호
- 자신이 속한 블록의 앞 블록까지의 스레드 수 + 자신이 속한 블록 내에서 자신의 번호
- NTHREAD_PER_BLK = blockDim.z * blockDim.y * blockDim.x
- 1차원 그리드: NTHREAD_PER_BLK * blockIdx.x + TID_IN_BLK
- 2차원 그리드: NTHREAD_PER_BLK * (gridDim.x * blockIdx.y + blockIdx.x) + TID_IN_BLK
- 3차원 그리드: NTHREAD_PER_BLK * (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) + TID_IN_BLK



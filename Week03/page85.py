#cnn 모델 정의
class FashionCNN(nn.Module):
    def __init__(self): #초기화 메서드
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            #필터 64개(서로 다른 64개의 특징을 찾기 위해)를 3x3 크기로 받고 stride는 1로 하여 한 번에 한 칸씩 이동한다.
            #패딩은 1로 설정하여 이미지 원본의 크기가 줄어들지 않도록 한다. 가장자리에 0으로 한칸씩 채운다.
            nn.ReLU(),
            #활성화 함수
            nn.MaxPool2d(kernel_size=2, stride=2))
            #2x2의 필터를 사용하여 4칸 중에 가장 숫자가 큰 것만 남긴다. 그리고 stride를 2로 하여 한 번에 두 칸씩 이동한다.
            #이미지의 크기가 1/4로 줄어든다. (28x28 -> 14x14)
        self.layer2 = nn.Sequential(
            #두번째 층이자 은닉층의 역할을 한다. 첫번째 층에서 추출한 특징을 바탕으로 더 복잡한 특징을 추출한다.
            nn.Conv2d(64, 64, kernel_size=3),
            #첫번째 층에서 받은 64개의 특징을 똑같이 64개의 특징을 잡는 필터 크기 3x3으로 하려 받는다.
            #이때 stride는 기본값 1, padding은 기본값 0으로 설정한다. (14x14 -> 12x12)
            nn.ReLU(),
            nn.MaxPool2d(2))
            #2x2의 필터를 사용하여 4칸 중에 가장 숫자가 큰 것만 남긴다. stride가 없다면 커널 사이즈와 같아진다. (12x12 -> 6x6)

        self.fc1 = nn.Linear(64*6*6, 128)
        #필터의 수는 64개이고 마지막으로 출력되는 이미지 맵의 크기는 6x6이므로 64*6*6개의 뉴런이 된다.
        #그리고 이를 128개의 뉴런으로 연결하는 선형 계층을 정의한다.
        self.fc2 = nn.Linear(128, 10)
        #128개를 입력받고 10개의 출력 뉴런으로 연결한다.
    

    def forward(self, x): #정방향 메서드
        out = self.layer1(x)
        #출력 형태: 배치 크기, 64채널, 14, 14
        out = self.layer2(out)
        #출력 형태: 배치 크기, 64채널, 6, 6
        out = out.view(out.size(0), -1)  # Flatten the tensor
        #out.size(0)는 그대로 유지하라는 뜻
        #64*6*6을 1차원으로 평탄화한다. -1은 알아서 하라는 뜻)
        out = self.fc1(out)
        #2,304개의 데이터를 선형 변환하여 128개로 축소
        out = self.fc2(out)
        #2,304개의 데이터를 선형 변환하여 128개로 축소
        return out
import numpy as np
import cv2

#função para remover centro do contorno
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

#Abre o vídeo/camera
cap = cv2.VideoCapture(0)

#Background para mascara 
fgbg = cv2.createBackgroundSubtractorMOG2()

detects = []

#posL Linha na vertical
posL = 220
#offset quantidade de pixel para cima ou para baixo para começar a contar 
offset = 30

#Posição da linha
xy1 = (20, posL)
xy2 = (600, posL)

total = 0

up = 0
down = 0

#loop para ler a camera/ arquivo de vídeo
while 1:
    #ret = retorno do vídeo || Frame é a foto do vídeo
    ret, frame = cap.read()

    #Converte o frame para cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    #Tirar a mascara. Diferença do frame anterior p/ próximo frame
    fgmask = fgbg.apply(gray)
    # cv2.imshow("fgmask", fgmask)

    #Trashold serve p/ remover noise da imagem 
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("th", th)

    #Kernel para usar morfologia do opencv || Remoção de noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    #morfologia para limpar a imagem                       em iterations que mexe para configurar de acordo a camera
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow("opening", opening)

    dilation = cv2.dilate(opening, kernel, iterations=10)
    # cv2.imshow("dilation", dilation)

    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=10)
    cv2.imshow("closing", closing)

    #Adicionando linhas
    cv2.line(frame, xy1, xy2, (255, 0, 0), 3)

    cv2.line(frame, (xy1[0], posL - offset), (xy2[0], posL - offset), (255, 255, 0), 2)

    cv2.line(frame, (xy1[0], posL + offset), (xy2[0], posL + offset), (255, 255, 0), 2)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
   # i serve para contagem
    i = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        #Verificando se o tamanho do objeto é o suficiente
        if int(area) > 3000: #TESTE
            centro = center(x, y, w, h)

            cv2.putText(frame, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            #Ponto vermelho no centro
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(detects) <= i:
                detects.append([])
            if centro[1] > posL - offset and centro[1] < posL + offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1

    if i == 0:
        detects.clear()

    i = 0

    if len(contours) == 0:
        detects.clear()

    else:

        for detect in detects:
            for (c, l) in enumerate(detect):

                if detect[c - 1][1] < posL and l[1] > posL:
                    detect.clear()
                    up += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 255, 0), 5)
                    continue

                if detect[c - 1][1] > posL and l[1] < posL:
                    detect.clear()
                    down += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                    continue

                if c > 0:
                    cv2.line(frame, detect[c - 1], l, (0, 0, 255), 1)

    cv2.putText(frame, "TOTAL: " + str(total), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, "SUBINDO: " + str(up), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "DESCENDO: " + str(down), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #Exibição
    cv2.imshow("frame", frame)

    #waitKey para dar um delay || tecle "q" para parar a aplicação
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#Liberar a tela
cap.release()
cv2.destroyAllWindows()
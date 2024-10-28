import cv2
import mediapipe as mp
import time

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Initialiser les solutions MediaPipe pour les mains
mpMains = mp.solutions.hands
mains = mpMains.Hands()
mpDessiner = mp.solutions.drawing_utils

# Identifiants pour les bouts des doigts
tipid = [4, 8, 12, 16, 20]
listePass = []

while True:
    # Lire l'image de la caméra
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Retourner l'image horizontalement

    # Convertir l'image BGR en RGB pour l'analyse
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultats = mains.process(imgRgb)
    
    # Liste pour stocker les coordonnées des points de repère
    listeLms = []
    if resultats.multi_hand_landmarks:
        for mainLms in resultats.multi_hand_landmarks:
            for id, lm in enumerate(mainLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Coordonnées des points de la main
                listeLms.append([id, cx, cy])
                mpDessiner.draw_landmarks(img, mainLms, mpMains.HAND_CONNECTIONS)  # Dessiner les connexions de la main
                
                if id == 8:  # Si c'est le bout du doigt index
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)  # Cercle vert au bout du doigt
                    
                if len(listeLms) == 21:
                    doigts = []
                    
                    # Vérifier si le pouce est levé
                    if listeLms[tipid[0]][1] < listeLms[tipid[0]-2][1]:
                        doigts.append(1)
                    else:
                        doigts.append(0)

                    # Vérifier les autres doigts
                    for i in range(1, 5):
                        if listeLms[tipid[i]][2] < listeLms[tipid[i]-2][2]:
                            doigts.append(1)
                        else:
                            doigts.append(0)
          
                    totalDoigts = doigts.count(1)  # Compter le nombre de doigts levés
                    print(totalDoigts)
                    
                    # Afficher le nombre de doigts levés à l'écran
                    cv2.putText(img, f'{totalDoigts}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    # Ajouter à la liste des doigts
                    listePass.append(totalDoigts)
                    ch = ''.join(str(x) for x in listePass)  # Convertir la liste en chaîne
                    cv2.putText(img, f'{ch}', (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Afficher l'image avec les annotations
    cv2.imshow("suivi des mains", img)
    
    # Quitter avec la touche 'Échap'
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

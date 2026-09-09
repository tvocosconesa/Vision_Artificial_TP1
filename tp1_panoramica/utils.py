from colorsys import hsv_to_rgb

import numpy as np 
from matplotlib import pyplot as plt
import cv2


def plot_matches(img1, kp1, img2, kp2, matches, titulo=None, figsize=(20, 10)):
    """Grafica cada match con un color distinto y circulos en sus keypoints."""
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, [], None,
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    for i, match in enumerate(matches):
        # El paso aureo separa los tonos de conexiones consecutivas.
        color = hsv_to_rgb((i * 0.618033988749895) % 1, 0.85, 1)
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        x2 += img1.shape[1]
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=1.2)
        plt.scatter(
            [x1, x2], [y1, y2], s=80, facecolors='none',
            edgecolors=[color], linewidths=2, zorder=3
        )
    plt.axis('off')
    plt.title(titulo if titulo else f'{len(matches)} matches encontrados')
    plt.show()


def detectar_matches(descr1, descr2, ratio=0.75, plot=False,
                      img1=None, kp1=None, img2=None, kp2=None, titulo=None):
    """Matchea dos sets de descriptores con BFMatcher + ratio test de Lowe.

    Si plot=True, ademas grafica los matches llamando a plot_matches
    (para lo cual hay que pasar img1, kp1, img2, kp2).
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descr1, descr2, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]

    print(f'Matches buenos: {len(good_matches)} de {len(matches)} totales')

    if plot:
        plot_matches(img1, kp1, img2, kp2, good_matches, titulo=titulo)

    return good_matches


def anms(keypoints, n_max):
    """
    Supresión No Máxima Adaptativa (ANMS).
    
    Selecciona los N keypoints que maximizan la dispersión espacial,
    priorizando aquellos con mayor 'respuesta' (response) que además
    estén rodeados de un radio libre de otros puntos más fuertes.
    
    Parámetros
    ----------
    keypoints : list[cv2.KeyPoint]
        Lista de keypoints detectados (ej. por SIFT, ORB, etc.)
    n_max : int
        Número máximo de keypoints a retener.
    
    Retorna
    -------
    list[cv2.KeyPoint]
        Los n_max keypoints seleccionados, bien distribuidos espacialmente.
    """
    n = len(keypoints)
    
    # Si ya tenemos menos keypoints que el máximo pedido, no hay nada que suprimir
    if n <= n_max:
        return keypoints
    
    # Extraer coordenadas (x, y) y respuestas (r) de cada keypoint
    coords = np.array([kp.pt for kp in keypoints])       # shape (n, 2)
    responses = np.array([kp.response for kp in keypoints])  # shape (n,)
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Matriz de distancias euclidianas al cuadrado entre todos los pares (i, j)
    # SD[i, j] = (x_j - x_i)^2 + (y_j - y_i)^2
    dx = x[np.newaxis, :] - x[:, np.newaxis]  # dx[i, j] = x_j - x_i
    dy = y[np.newaxis, :] - y[:, np.newaxis]  # dy[i, j] = y_j - y_i
    SD = dx**2 + dy**2                         # shape (n, n)
    
    # Matriz de condición: r_j > r_i  (solo miramos vecinos "más fuertes")
    resp_i = responses[:, np.newaxis]  # r_i, shape (n, 1)
    resp_j = responses[np.newaxis, :]  # r_j, shape (1, n)
    mask_stronger = resp_j > resp_i    # True donde r_j > r_i
    
    # Donde la condición no se cumple, invalidamos esa distancia (poniendo infinito)
    # así no interfiere al calcular el mínimo
    SD_masked = np.where(mask_stronger, SD, np.inf)
    
    # R_i = distancia mínima al vecino más cercano que sea más fuerte que i
    R = np.min(SD_masked, axis=1)
    
    # Si un keypoint no tiene ningún vecino más fuerte (ej. el máximo global),
    # R_i queda en infinito -> lo dejamos así para que siempre quede primero
    # al ordenar de forma descendente
    
    # Ordenar de forma descendente por R_i y quedarnos con los N mejores
    order = np.argsort(-R)  # -R para orden descendente
    top_indices = order[:n_max]
    
    selected_keypoints = [keypoints[i] for i in top_indices]
    
    return selected_keypoints

def graf_keypoints(img,keypoints,r=0):
    automatico=0

    if(r==0):
        automatico=1
    
    img_copy=img.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if(automatico==1):
            r = max(7, int(kp.size / 3))
        cv2.circle(img_copy, (x, y), r + 2, (0, 0, 0), -1)    # halo negro
        cv2.circle(img_copy, (x, y), r, (0, 0, 255), -1)      # rojo llamativo (BGR)
    return img_copy

def matchea(desc1,desc2,trees=10,checks=50):
    dict_indices=dict(algorithm=1, trees=trees)
    search_params = dict(checks=checks)
    #nuestro "matcher"
    flann = cv2.FlannBasedMatcher(dict_indices,search_params)
    #conseguimos los matches:
    matches_k2 = flann.knnMatch(desc1, desc2, k=2)

    #Mejores matches:
    buenos_matches = []
    for m, n in matches_k2:
        # Si la distancia al 1er vecino es mucho menor que al 2do vecino, es un buen match
        if m.distance < 0.75 * n.distance:
            buenos_matches.append(m)
    return buenos_matches

def calc_anms(img,_print:bool=False):
    sift = cv2.SIFT_create()  # sin limitar nfeatures, detectamos todos primero
    all_keys, all_desc = sift.detectAndCompute(img, None)
    kp_anms = anms(all_keys, n_max=200)

    if(_print):
        print(f'Keypoints detectados originalmente: {len(all_keys)}')
        print(f'Keypoints tras ANMS: {len(kp_anms)}')
    
    return all_keys,all_desc,kp_anms

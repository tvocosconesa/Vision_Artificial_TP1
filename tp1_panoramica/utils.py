import numpy as np 
from matplotlib import pyplot as plt
import cv2


def plot_matches(img1, kp1, img2, kp2, matches, titulo=None, figsize=(20, 10)):
    """Grafica los matches entre dos imagenes usando cv2.drawMatches."""
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
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
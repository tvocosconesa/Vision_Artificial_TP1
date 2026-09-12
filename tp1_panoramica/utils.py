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


def plot_point_correspondences(img1, pts1, img2, pts2, titulo=None, figsize=(20, 10)):
    """Grafica pares de puntos seleccionados manualmente entre dos imagenes,
    numerando cada par y coloreandolo igual en ambos lados para poder
    identificar visualmente la correspondencia."""
    h1, w1 = img1.shape[:2]
    canvas = np.concatenate([
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    ], axis=1)

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        color = hsv_to_rgb((i * 0.618033988749895) % 1, 0.85, 1)
        x2_shifted = x2 + w1
        plt.plot([x1, x2_shifted], [y1, y2], color=color, linewidth=1.5, linestyle='--')
        plt.scatter([x1, x2_shifted], [y1, y2], s=150, facecolors='none',
                    edgecolors=[color], linewidths=2.5, zorder=3)
        plt.text(x1, y1 - 15, str(i + 1), color=color, fontsize=14, fontweight='bold', ha='center')
        plt.text(x2_shifted, y2 - 15, str(i + 1), color=color, fontsize=14, fontweight='bold', ha='center')
    plt.axis('off')
    plt.title(titulo if titulo else f'{len(pts1)} correspondencias seleccionadas')
    plt.show()


def plot_warp_result(img_src, img_dst, H, titulo=None, figsize=(12, 8)):
    """Aplica una homografia H a img_src y la mezcla (50/50) con img_dst
    para verificar visualmente que tan bien queda alineada la transformacion."""
    h, w = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h))
    blend = cv2.addWeighted(warped, 0.5, img_dst, 0.5, 0)

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(titulo if titulo else 'Resultado del warping (mezcla 50/50 con el ancla)')
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

    Retorna
    -------
    selected_keypoints : list[cv2.KeyPoint]
        Los n_max keypoints seleccionados, bien distribuidos espacialmente.
    top_indices : np.ndarray (n_max,)
        Índices de los keypoints seleccionados dentro de la lista original.
        Necesarios para filtrar el array de descriptores en paralelo y
        mantener la correspondencia keypoint[i] <-> descriptor[i].
    """
    n = len(keypoints)

    # Si ya tenemos menos keypoints que el máximo pedido, no hay nada que suprimir
    if n <= n_max:
        return list(keypoints), np.arange(n)

    coords = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    x = coords[:, 0]
    y = coords[:, 1]

    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    SD = dx**2 + dy**2

    resp_i = responses[:, np.newaxis]
    resp_j = responses[np.newaxis, :]
    mask_stronger = resp_j > resp_i

    SD_masked = np.where(mask_stronger, SD, np.inf)
    R = np.min(SD_masked, axis=1)

    order = np.argsort(-R)
    top_indices = order[:n_max]

    selected_keypoints = [keypoints[i] for i in top_indices]

    return selected_keypoints, top_indices

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

def calc_anms(img, n_max=200, _print=False):
    """Detecta keypoints con SIFT, aplica ANMS y devuelve los keypoints
    seleccionados junto con SUS descriptores correspondientes (alineados
    índice a índice, que es lo que necesita el matching)."""
    sift = cv2.SIFT_create()  # sin limitar nfeatures, detectamos todos primero
    all_keys, all_desc = sift.detectAndCompute(img, None)

    kp_anms, idx_anms = anms(all_keys, n_max=n_max)
    desc_anms = all_desc[idx_anms]   # <-- la línea que faltaba

    if _print:
        print(f'Keypoints detectados originalmente: {len(all_keys)}')
        print(f'Keypoints tras ANMS: {len(kp_anms)}')

    return kp_anms, desc_anms

def _normalizar_puntos(pts):
    pts = np.asarray(pts, dtype=np.float64)
    centroide = pts.mean(axis=0)
    desplazados = pts - centroide
    dist_media = np.mean(np.sqrt(np.sum(desplazados ** 2, axis=1)))

    escala = np.sqrt(2) / dist_media if dist_media > 1e-8 else 1.0

    T = np.array([
        [escala, 0,      -escala * centroide[0]],
        [0,      escala, -escala * centroide[1]],
        [0,      0,       1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def calcular_homografia_dlt(pts_origen, pts_destino, normalizar=True):
    pts_origen = np.asarray(pts_origen, dtype=np.float64)
    pts_destino = np.asarray(pts_destino, dtype=np.float64)

    if pts_origen.shape[0] < 4 or pts_destino.shape[0] < 4:
        raise ValueError("Se necesitan al menos 4 correspondencias de puntos.")
    if pts_origen.shape != pts_destino.shape:
        raise ValueError("pts_origen y pts_destino deben tener la misma forma.")

    if normalizar:
        pts_o, T_o = _normalizar_puntos(pts_origen)
        pts_d, T_d = _normalizar_puntos(pts_destino)
    else:
        pts_o, pts_d = pts_origen, pts_destino
        T_o = T_d = np.eye(3)

    # Construir la matriz A (2N x 9) a partir de x' x (H x) = 0
    filas = []
    for (x, y), (xp, yp) in zip(pts_o, pts_d):
        filas.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        filas.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(filas)

    # Resolver A h = 0: h es el vector singular derecho asociado
    # al menor valor singular (ultima fila de Vt en la SVD de A)
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Deshacer la normalizacion de Hartley
    H = np.linalg.inv(T_d) @ H_norm @ T_o

    # Fijar la escala para que H[2,2] = 1
    H = H / H[2, 2]
    return H


def homografia_entre_imagenes(img1, img2, pts1, pts2, normalizar=True):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    for (x, y) in pts1:
        if not (0 <= x < w1 and 0 <= y < h1):
            raise ValueError(f"Punto {(x, y)} fuera de los limites de img1 ({w1}x{h1}).")
    for (x, y) in pts2:
        if not (0 <= x < w2 and 0 <= y < h2):
            raise ValueError(f"Punto {(x, y)} fuera de los limites de img2 ({w2}x{h2}).")

    return calcular_homografia_dlt(pts1, pts2, normalizar=normalizar)

def _proyectar_puntos(H, pts):
    """Aplica H a un conjunto de puntos 2D (Nx2) usando coordenadas
    homogeneas y devuelve los puntos resultantes ya divididos por w."""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proy_h = (H @ pts_h.T).T
    w = proy_h[:, 2]
    # Evitar division por cero (puntos que se van al infinito con una H degenerada)
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return proy_h[:, :2] / w[:, None]
 
def ransac_homografia(k_i, k_j, T=1000, t=3.0, semilla=None):
    """
    Estima la homografia H que mapea k_i -> k_j de forma robusta ante
    correspondencias incorrectas (outliers), usando RANSAC.
 
    Parametros
    ----------
    k_i : array-like (N,2)
        Keypoints detectados en la imagen I1.
    k_j : array-like (N,2)
        Keypoints correspondientes detectados en la imagen I2 (mismo orden
        que k_i, es decir k_i[n] <-> k_j[n]).
    T : int
        Cantidad de iteraciones de RANSAC a realizar.
    t : float
        Umbral de distancia (en pixeles) para considerar una correspondencia
        como inlier.
    semilla : int o None
        Semilla del generador aleatorio, para resultados reproducibles.
 
    Retorna
    -------
    H : ndarray (3,3)
        Homografia final, recalculada con cuadrados minimos (DLT) usando
        todas las correspondencias inliers del mejor modelo encontrado.
    inliers : ndarray de bools, forma (N,)
        Mascara indicando que correspondencias de (k_i, k_j) son inliers
        respecto a H.
    """
    k_i = np.asarray(k_i, dtype=np.float64)
    k_j = np.asarray(k_j, dtype=np.float64)
 
    n = k_i.shape[0]
 
    rng = np.random.default_rng(semilla)
    indices = np.arange(n)
 
    mejor_inliers = None
    mejor_num_inliers = -1
 
    # 1: for i = [1:T] do
    for _ in range(T):
        # 2: Seleccionar 4 pares de correspondencias aleatorias
        muestra = rng.choice(indices, size=4, replace=False)
 
        # 3: Calcular homografia H utilizando los pares seleccionados
        try:
            H_muestra = calcular_homografia_dlt(k_i[muestra], k_j[muestra])
        except np.linalg.LinAlgError:
            # 4 puntos degenerados (colineales, repetidos, etc.) -> descartar
            continue
 
        # 4: Determinar inliers tal que dist(k_j, H*k_i) < t
        proyectados = _proyectar_puntos(H_muestra, k_i)
        distancias = np.sqrt(np.sum((k_j - proyectados) ** 2, axis=1))
        inliers = distancias < t
        num_inliers = int(np.sum(inliers))
 
        # 5: Recordar el conjunto de inliers mas grande
        if num_inliers > mejor_num_inliers:
            mejor_num_inliers = num_inliers
            mejor_inliers = inliers
 
    # 7: Recalcular H con cuadrados minimos utilizando todos los inliers
    H_final = calcular_homografia_dlt(k_i[mejor_inliers], k_j[mejor_inliers])
 
    return H_final, mejor_inliers


def plot_point_correspondences(img1, pts1, img2, pts2, titulo=None, figsize=(20, 10)):
    """Grafica pares de puntos seleccionados manualmente entre dos imagenes,
    numerando cada par y coloreandolo igual en ambos lados para poder
    identificar visualmente la correspondencia."""
    h1, w1 = img1.shape[:2]
    canvas = np.concatenate([
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    ], axis=1)

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        color = hsv_to_rgb((i * 0.618033988749895) % 1, 0.85, 1)
        x2_shifted = x2 + w1
        plt.plot([x1, x2_shifted], [y1, y2], color=color, linewidth=1.5, linestyle='--')
        plt.scatter([x1, x2_shifted], [y1, y2], s=150, facecolors='none',
                    edgecolors=[color], linewidths=2.5, zorder=3)
        plt.text(x1, y1 - 15, str(i + 1), color=color, fontsize=14, fontweight='bold', ha='center')
        plt.text(x2_shifted, y2 - 15, str(i + 1), color=color, fontsize=14, fontweight='bold', ha='center')
    plt.axis('off')
    plt.title(titulo if titulo else f'{len(pts1)} correspondencias seleccionadas')
    plt.show()


def plot_warp_result(img_src, img_dst, H, titulo=None, figsize=(12, 8)):
    """Aplica una homografia H a img_src y la mezcla (50/50) con img_dst
    para verificar visualmente que tan bien queda alineada la transformacion."""
    h, w = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h))
    blend = cv2.addWeighted(warped, 0.5, img_dst, 0.5, 0)

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(titulo if titulo else 'Resultado del warping (mezcla 50/50 con el ancla)')
    plt.show()
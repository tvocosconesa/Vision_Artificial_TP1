from graficos import *

#------------------------------------------
#A-NMS
def calc_anms(img, n_max=200, _print=False):
    """Detecta keypoints con SIFT, aplica ANMS y devuelve los keypoints
    seleccionados junto con SUS descriptores correspondientes (alineados
    índice a índice, que es lo que necesita el matching)."""
    sift = cv2.SIFT_create(nfeatures=16000)  # sin limitar nfeatures, detectamos todos primero
    all_keys, all_desc = sift.detectAndCompute(img, None)

    kp_anms, idx_anms = anms(all_keys, n_max=n_max)
    desc_anms = all_desc[idx_anms]   

    if _print:
        print(f'Keypoints detectados originalmente: {len(all_keys)}')
        print(f'Keypoints tras ANMS: {len(kp_anms)}')

    return kp_anms, desc_anms

def anms(keypoints, n_max):
    """
    Supresión No Máxima Adaptativa (ANMS).

    Selecciona los N keypoints que maximizan la dispersión espacial,
    priorizando aquellos con mayor 'respuesta' (response) que además
    estén rodeados de un radio libre de otros puntos más fuertes.

    Retorna=
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

#-------------------------------------
#Matching

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

def crosscheck_Lowe_Matching(desc1, desc2, threshold = 0.75):
    """
        Encuentra correspondencias confiables entre dos conjuntos de descriptores
        combinando dos criterios de filtrado: ratio test de Lowe y verificación
        cruzada (cross-check).

        El matching se realiza en ambas direcciones (desc1 -> desc2 y desc2 -> desc1)
        con busqueda exhaustiva (BFMatcher, norma L2), que es exacta y determinista.
        En cada dirección se aplica primero el ratio test de Lowe para
        descartar matches ambiguos (donde el 1er y 2do vecino más cercano están a
        distancias similares). Luego, de los matches que sobreviven el ratio test en
        ambas direcciones, se conservan solo aquellos que son mutuamente el mejor
        match del otro (cross-check), es decir, correspondencias simétricas.

        Parameters
        ----------
        desc1 : np.ndarray
            Descriptores de la imagen 1 (consulta), de forma (N1, D).
        desc2 : np.ndarray
            Descriptores de la imagen 2 (candidatos), de forma (N2, D).
        threshold : float, optional
            Umbral del ratio test de Lowe. Un match se acepta si distance(1er vecino)
            < threshold * distance(2do vecino) (default 0.75).

        Returns
        -------
        list[cv2.DMatch]
            Lista de matches que pasan tanto el ratio test como la verificación
            cruzada en ambas direcciones. Cada DMatch tiene queryIdx referido a
            desc1 y trainIdx referido a desc2.
    """

    bf = cv2.BFMatcher(cv2.NORM_L2)

    good_matches_12 = []
    good_matches_21 = []

    # Dirección 1 -> 2 (mejor par de vecinos, k=2)
    matches_12 = bf.knnMatch(desc1, desc2, k=2)

    for m, n in matches_12:
        if m.distance < threshold * n.distance:
            good_matches_12.append(m)

    # Dirección 2 -> 1
    matches_21 = bf.knnMatch(desc2, desc1, k=2)

    for m, n in matches_21:
        if m.distance < threshold * n.distance:
            good_matches_21.append(m)

    # Para verificar simetría rápido, armamos un diccionario:
    # para cada query en 2, cuál es su mejor match en 1
    mejor_de_2_en_1 = {m.queryIdx: m.trainIdx for m in good_matches_21 }

    final_good_matches = []
    for match in good_matches_12:
        if mejor_de_2_en_1.get(match.trainIdx) == match.queryIdx:
            final_good_matches.append(match)

    return final_good_matches

#-------------------------------------
#Cálculos Homografia

def estimar_homografia(kp0, desc0, kp1, desc1, t=5.0, T=1000, semilla=0, _print=False):
    """
    Realiza el matching cruzado entre descriptores, filtra con RANSAC 
    y devuelve la homografía final junto con los matches válidos (inliers).
    """
    
    matches = crosscheck_Lowe_Matching(desc0, desc1)
    
    #Validación de seguridad para evitar que RANSAC colapse
    if len(matches) < 4:
        raise ValueError(f"Insuficientes matches: Se encontraron {len(matches)} y se necesitan al menos 4.")

    #Extrae (x, y) de los keypoints emparejados
    pts0 = np.array([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.array([kp1[m.trainIdx].pt for m in matches])

    H, inliers_mask = ransac_homografia(pts0, pts1, t=t, T=T, semilla=semilla)
    #Filtrar la lista de objetos DMatch conservando solo los inliers
    inlier_matches = [m for m, es_inlier in zip(matches, inliers_mask) if es_inlier]
    
    if _print:
        print(f'Matches iniciales con cross-check: {len(matches)}')
        print(f'Inliers RANSAC: {len(inlier_matches)} de {len(matches)}')

    return H, inlier_matches

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

#Cálculos para realizar las panorámicas de las imágenes

def calcular_lienzo_panoramica(shape_izq, shape_centro, shape_der, H_izq, H_der):
    """
    Calcula las dimensiones totales del lienzo y la matriz de traslación T
    para evitar que las imágenes laterales se recorten.
    """
    h_i, w_i = shape_izq[:2]
    h_c, w_c = shape_centro[:2]
    h_d, w_d = shape_der[:2]

    # esquinas 
    corners_izq = np.float32([[0, 0], [0, h_i], [w_i, h_i], [w_i, 0]]).reshape(-1, 1, 2)
    corners_centro = np.float32([[0, 0], [0, h_c], [w_c, h_c], [w_c, 0]]).reshape(-1, 1, 2)
    corners_der = np.float32([[0, 0], [0, h_d], [w_d, h_d], [w_d, 0]]).reshape(-1, 1, 2)

    # Proyectar las esquinas laterales al espacio central
    corners_izq_warped = cv2.perspectiveTransform(corners_izq, H_izq)
    corners_der_warped = cv2.perspectiveTransform(corners_der, H_der)

    # Unir todas las esquinas y buscar límites globales
    all_corners = np.concatenate((corners_centro, corners_izq_warped, corners_der_warped), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Definir tamaño y matriz de traslación
    new_width = x_max - x_min
    new_height = y_max - y_min
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    return T, (new_width, new_height)


def proyectar_vistas(img_izq, img_centro, img_der, H_izq, H_der, T, size):
    """
    Aplica las transformaciones geométricas (Homografía + Traslación) a las 3 imágenes.
    """
    warped_izq = cv2.warpPerspective(img_izq, T @ H_izq, size)
    warped_centro = cv2.warpPerspective(img_centro, T, size)
    warped_der = cv2.warpPerspective(img_der, T @ H_der, size)
    
    return warped_izq, warped_centro, warped_der


def aplicar_blending_suave(warped_izq, warped_centro, warped_der):
    """
    Calcula los pesos mediante Distance Transform y fusiona las imágenes
    sin dejar costuras duras.
    """
    # 1. Máscaras binarias
    mask_izq = (cv2.cvtColor(warped_izq, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask_centro = (cv2.cvtColor(warped_centro, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask_der = (cv2.cvtColor(warped_der, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

    # 2. Mapas de distancia
    dist_izq = cv2.distanceTransform(mask_izq, cv2.DIST_L2, 3)
    dist_centro = cv2.distanceTransform(mask_centro, cv2.DIST_L2, 3)
    dist_der = cv2.distanceTransform(mask_der, cv2.DIST_L2, 3)

    # 3. Suma y normalización (Pesos Alpha)
    dist_sum = dist_izq + dist_centro + dist_der
    dist_sum[dist_sum == 0] = 1.0  # Evitar división por cero
    
    alpha_izq = np.dstack([dist_izq / dist_sum] * 3)
    alpha_centro = np.dstack([dist_centro / dist_sum] * 3)
    alpha_der = np.dstack([dist_der / dist_sum] * 3)

    # 4. Mezcla ponderada final
    blended = (warped_izq * alpha_izq + warped_centro * alpha_centro + warped_der * alpha_der).astype(np.uint8)
    return blended


def plot_warp_3_imagenes(img_izq, img_centro, img_der, H_izq, H_der, titulo=None, figsize=(20, 10)):
    """
    grafica la panorámica.
    """
    # 1. Calcular límites
    T, size = calcular_lienzo_panoramica(img_izq.shape, img_centro.shape, img_der.shape, H_izq, H_der)
    
    # 2. Proyectar
    w_izq, w_centro, w_der = proyectar_vistas(img_izq, img_centro, img_der, H_izq, H_der, T, size)
    
    # 3. Fusionar
    blended = aplicar_blending_suave(w_izq, w_centro, w_der)

    # 4. Graficar
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(titulo if titulo else 'Panorámica Final (Modularizada)')
    plt.show()

    return blended


#----------------------------------------------
#Utils

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

def calcular_error_reproyeccion(H, kp_src, kp_dst, inlier_matches):
    """
    Calcula el error de reproyección promedio (en píxeles) para un conjunto de inliers.
    """
  
    pts_src = np.array([kp_src[m.queryIdx].pt for m in inlier_matches], dtype=np.float64)
    pts_dst = np.array([kp_dst[m.trainIdx].pt for m in inlier_matches], dtype=np.float64)

    pts_proyectados = _proyectar_puntos(H, pts_src)

    #Calcula la distancia euclidiana entre lo real y la proyección matemática
    distancias = np.sqrt(np.sum((pts_dst - pts_proyectados) ** 2, axis=1))

    error_medio = np.mean(distancias)
    
    return error_medio
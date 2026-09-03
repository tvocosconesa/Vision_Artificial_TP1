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
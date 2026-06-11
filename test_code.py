"""
def function_chooser(func_type):

    Takes a string as a function type,
    modifies X and/or Y as needed, then
    returns X, Y, and n.

    fit_type = func_type

    # fit_type = "linear"
    if fit_type == "polynomial":
        n = int(
            input("What degree of polynomial? Please give your answer as a numeral.")
        )
    elif fit_type == "exponential":
        Y = np.log(Y)
        n = 1
    elif fit_type == "power":
        Y = np.log(Y)
        X = np.log(X)
        n = 1
    elif fit_type == "linear":
        n = 1
    else:
        print("Request cannot be completed, defaulting to rat")
        import matplotlib.image as mpimg

        img = mpimg.imread("fat_rats/Joanna_Servaes_wikimedia_commons.webp")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        fit_type = None
        return
"""

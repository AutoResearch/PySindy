import matplotlib.pyplot as plt


class Model:

    def __init__(self):
        self.t = None
        self.y = None
        self.dim = 0

    def visualize_model(self):
        if self.dim == 0:
            raise Warning("Model Uninitialized")
        elif self.dim == 1:
            plt.figure(figsize=(8, 4))
            plt.xlabel("Time")
            plt.ylabel("")
            for i in range(self.y.shape[0]):
                plt.plot(self.t, self.y[i,:])
            plt.show()
        elif self.dim == 2:
            plt.figure(figsize=(8, 4))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(self.y[0], self.y[1])
            plt.show()
        elif self.dim == 3:
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(self.y[0, :], self.y[1, :], self.y[2, :], 'k')
            ax1.set(xlabel='$x_0$', ylabel='$x_1$',
                    zlabel='$x_2$')
            plt.show()


class Model1D(Model):

    def __init__(self):
        Model.__init__(self)
        self.dim = 1


class Model2D(Model):

    def __init__(self):
        Model.__init__(self)
        self.dim = 2


class Model3D(Model):

    def __init__(self):
        Model.__init__(self)
        self.dim = 3

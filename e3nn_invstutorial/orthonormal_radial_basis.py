import math
import torch


def FixedCosineRadialModel(max_radius, number_of_basis, min_radius=0.):
    spacing = (max_radius - min_radius) / number_of_basis
    radii = torch.linspace(min_radius, max_radius, number_of_basis)
    step = radii[1] - radii[0]

    def radial_function(r):
        shape = r.shape
        radial_shape = [1] * len(shape) + [number_of_basis]
        centers = radii.reshape(*radial_shape)
        return (r.unsqueeze(-1) - centers).div(step).add(1).relu().sub(2).neg().relu().add(1).mul(
            math.pi / 2).cos().pow(2)

    return radial_function


class CosineFunctions:
    def __init__(self, nRadialFunctions, radialCutoff):
        self.nRadialFunctions = nRadialFunctions
        self.radialCutoff = radialCutoff
        self.factors = (math.pi / radialCutoff) * torch.arange(0, nRadialFunctions)

    def __call__(self, r):
        return torch.cos(torch.outer(r, self.factors))


# Have value and 1st derivative both go to 0 at the cutoff.
class FadeAtCutoff:
    def __init__(self, radialModel, radialCutoff):
        self.radialModel = radialModel
        self.radialCutoff = radialCutoff

    def __call__(self, r):
        f = self.radialModel(r)
        nRadialFunctions = f.shape[1]
        fadeFn = (self.radialCutoff - r) * (self.radialCutoff - r)
        fadeFn = torch.outer(fadeFn, torch.ones((nRadialFunctions,)))
        return f * fadeFn


# Orthonormalizes a set of radial basis functions on a sphere.
#   Uses modified Gram-Schmidt w/trapezoidal integration rule.
#   Tabulates radial basis, returns linear interpolation differentiable in r
class OrthonormalRadialFunctions:
    def innerProduct(self, a, b):
        return torch.trapz(a * b * self.areaSamples, self.radialSamples)

    def norm(self, a):
        return torch.sqrt(self.innerProduct(a, a))

    def __init__(self, num_radials, radialModel, rcut, num_samples):
        self.nRadialFunctions = num_radials
        self.radialCutoff = rcut

        self.radialSamples = torch.linspace(0, rcut, num_samples)
        self.radialStep = self.radialSamples[1] - self.radialSamples[0]

        nonOrthogonalSamples = radialModel(self.radialSamples)

        self.areaSamples = 4 * math.pi * self.radialSamples * self.radialSamples

        self.fSamples = torch.zeros_like(nonOrthogonalSamples)

        u0 = nonOrthogonalSamples[:, 0]
        self.fSamples[:, 0] = u0 / self.norm(u0)

        for i in range(1, num_radials):
            ui = nonOrthogonalSamples[:, i]
            for j in range(i):
                uj = self.fSamples[:, j]
                ui -= self.innerProduct(uj, ui) / self.innerProduct(uj, uj) * uj
            self.fSamples[:, i] = ui / self.norm(ui)

        self.radialStep

    def __call__(self, r):
        rNormalized = r / self.radialStep
        rNormalizedFloor = torch.floor(rNormalized)
        rNormalizedFloorInt = rNormalized.long()
        indicesLow = torch.min(torch.max(rNormalizedFloorInt, torch.tensor([0], dtype=torch.long)),
                               torch.tensor([len(self.radialSamples) - 2], dtype=torch.long))
        rRemainderNormalized = rNormalized - indicesLow
        rRemainderNormalized = torch.unsqueeze(rRemainderNormalized, -1)  # add a dimension at the end
        rRemainderNormalized = rRemainderNormalized.expand(
            list(rRemainderNormalized.shape[:-1]) + [self.nRadialFunctions])
        # rRemainderNormalized = torch.outer(rNormalized - indicesLow, torch.ones((self.nRadialFunctions,)))

        lowSamples = self.fSamples[indicesLow, :]
        highSamples = self.fSamples[indicesLow + 1, :]

        ret = lowSamples * (1 - rRemainderNormalized) + highSamples * rRemainderNormalized

        return ret


if __name__ == "__main__":
    radialCutoff = 3.5
    nRadialFunctions = 14

    fixedCosineRadialModel = FixedCosineRadialModel(radialCutoff, nRadialFunctions)
    cosineModel = CosineFunctions(nRadialFunctions, radialCutoff)
    cosineModelFaded = FadeAtCutoff(cosineModel, radialCutoff)

    r = torch.linspace(0, radialCutoff, 15)
    y1 = fixedCosineRadialModel(r)
    y2 = cosineModel(r)
    y3 = cosineModelFaded(r)

    onRadialModel = OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, radialCutoff, 100)
    r = torch.linspace(0, radialCutoff, 51, requires_grad=True)
    y = onRadialModel(r)
    g = torch.zeros_like(y)
    g[:, -1] += 1
    g2 = y.backward(gradient=g)

    torch.set_printoptions(linewidth=10000)
    print(y)
    print(r.grad)

    print(y.shape)

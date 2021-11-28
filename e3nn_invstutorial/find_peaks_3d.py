import torch
import torch.nn


def findPeaks3d(signalOnGrid, cutoff):
    peakGrid = (signalOnGrid >= cutoff)

    peakGrid[:-1, :, :] &= (signalOnGrid[:-1, :, :] > signalOnGrid[1:, :, :])
    peakGrid[1:, :, :] &= (signalOnGrid[1:, :, :] > signalOnGrid[:-1, :, :])

    peakGrid[:, :-1, :] &= (signalOnGrid[:, :-1, :] > signalOnGrid[:, 1:, :])
    peakGrid[:, 1:, :] &= (signalOnGrid[:, 1:, :] > signalOnGrid[:, :-1, :])

    peakGrid[:, :, :-1] &= (signalOnGrid[:, :, :-1] > signalOnGrid[:, :, 1:])
    peakGrid[:, :, 1:] &= (signalOnGrid[:, :, 1:] > signalOnGrid[:, :, :-1])

    indicesX, indicesY, indicesZ = peakGrid.nonzero(as_tuple=True)

    values = signalOnGrid[indicesX, indicesY, indicesZ]

    return (indicesX, indicesY, indicesZ, values)


if __name__ == "__main__":
    from tutorial.orthonormal_radial_basis import (
        OrthonormalRadialFunctions,
        FixedCosineRadialModel,
        CosineFunctions,
        FadeAtCutoff
    )

    from tutorial.radial_spherical_tensor import RadialSphericalTensor


    def orthogonalityTest(basisSamples, integrationWeights):
        nBases = basisSamples.shape[-1]
        orthogonalityCheck = torch.einsum('...b,...c,...->bc', basisSamples, basisSamples, integrationWeights)
        orthonormalityError = torch.max(torch.abs(orthogonalityCheck - torch.eye(nBases)))
        orthogonalityCheck2 = orthogonalityCheck / torch.mean(torch.diag(orthogonalityCheck))
        orthogonalityError = torch.max(torch.abs(orthogonalityCheck2 - torch.eye(nBases)))
        return (orthonormalityError, orthogonalityError, orthogonalityCheck)


    radialCutoff = 3.5
    lmax = 11
    p_val = 1
    p_arg = 1
    nRadialFunctions = (lmax + 1)

    fixedCosineRadialModel = FixedCosineRadialModel(radialCutoff, nRadialFunctions)
    cosineModel = CosineFunctions(nRadialFunctions, radialCutoff)
    cosineModelFaded = FadeAtCutoff(cosineModel, radialCutoff)
    onRadialModel = OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, radialCutoff, 1024)

    rst = RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    peakPoints = torch.rand((3, 3)) - 0.5
    peakPoints /= torch.linalg.norm(peakPoints)
    peakPoints *= 2

    signal = rst.with_peaks_at(peakPoints)
    linearSamples, signalOnGrid = rst.signal_on_grid(signal, radialCutoff, 100, cropBases=True, cutoffRadiusInner=None,
                                                     useCache=True)

    indicesX, indicesY, indicesZ, values = findPeaks3d(signalOnGrid, 0.5)
    peaksX = linearSamples[indicesX]
    peaksY = linearSamples[indicesY]
    peaksZ = linearSamples[indicesZ]

    print(torch.transpose(peakPoints, 0, 1))
    print(peaksX)
    print(peaksY)
    print(peaksZ)
    print('done')

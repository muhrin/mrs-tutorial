import math
import numpy as np
import torch

from e3nn import o3


class RadialSphericalTensor(o3.Irreps):
    r"""representation of a signal in 3-space or in a solid ball

    A 'RadialSphericalTensor' contains the coefficients of a function expansion in 3-space, potentially compactly supported on a solid ball.
    Each coefficient corresponds to a single basis function; each basis function is the product of a radial basis function and a single spherical harmonic.

    Arguments:

    nRadialBases: int>0, number of radial basis functions
    orthonormalRadialBases: a function or functional that accepts a vector of nR>0 radii,
        and returns an array of shape (nR,nRadialBases) containing the values of
        the orthonormal radial basis functions.
    lMax: int, the maximum degree of spherical harmonic functions
    p_val, p_arg: same as in SphericalTensor
    """

    def __new__(cls,
                num_radials, basis,  # Provide an orthonormal radial basis
                lmax, p_val, p_arg):  # provide an angular basis

        cls.num_radials = num_radials
        cls.radialBases = basis
        cls.lmax = lmax
        cls.p_val = p_val
        cls.p_arg = p_arg
        # cls.orthonormal = (orthonormal != 0)
        # cls.gridBasisFunctionCache = {}
        # if radialAngularSelector==None:
        #     cls.radialAngularSelector = None
        multiplicities = [num_radials] * (lmax + 1)
        # elif isinstance(radialAngularSelector, list):
        #     raise 'This is not yet implemented.'
        # else:
        #     cls.radialAngularSelector = np.array(radialAngularSelector, dtype=np.int8)
        #     multiplicities = np.count_nonzero(cls.radialAngularSelector, axis=0)

        radialSelector = []
        for l in range(lmax + 1):
            nm = 2 * l + 1
            for iRadial in range(num_radials):
                for m in range(nm):
                    radialSelector.append(iRadial)
        cls.radialSelector = torch.tensor(radialSelector)

        parities = {l: (p_val * p_arg ** l) for l in range(lmax + 1)}

        irreps = [(multiplicity, (l, parities[l])) for multiplicity, l in zip(multiplicities, range(lmax + 1))]
        ret = super().__new__(cls, irreps)

        return ret

    def _evaluateAngularBasis(self, vectors, radii=None):
        r"""Evaluate angular basis functions (spherical harmonics) at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        assert self[0][
                   1].p == 1, "the spherical harmonics are only evaluable when p_val is 1, since the l=0 must have parity 1."  # pylint: disable=no-member

        if radii is not None:
            angularCoeffs = o3.spherical_harmonics(self, vectors.view(-1, 3) / radii.view(-1, 1).expand(-1, 3),
                                                   normalize=False, normalization='integral') * 2 * math.sqrt(math.pi)
        else:
            angularCoeffs = o3.spherical_harmonics(self, vectors.view(-1, 3), normalize=True,
                                                   normalization='integral') * 2 * math.sqrt(math.pi)

        finalShape = tuple(list(vectors.shape[:-1]) + [self.dim])
        basisValuesNotFlat = angularCoeffs.view(finalShape)

        return basisValuesNotFlat

    def _evaluateRadialBasis(self, vectors, radii=None):
        r"""Evaluate radial basis functions at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        if radii is not None:
            basesFlat = self.radialBases(radii.view(-1))
        else:
            basesFlat = self.radialBases(torch.norm(vectors, dim=-1).view(-1))

        basesFlat = basesFlat[:, self.radialSelector]
        finalShape = tuple(list(vectors.shape[:-1]) + [self.dim])

        basisValuesNotFlat = basesFlat.view(finalShape)

        return basisValuesNotFlat

    def _evaluateJointBasis(self, vectors, radii=None):
        r"""Evaluate joint (radial x angular) basis functions at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        radii = torch.norm(vectors, dim=-1)
        angularBasis = self._evaluateAngularBasis(vectors, radii)
        radialBasis = self._evaluateRadialBasis(vectors, radii)
        return (angularBasis * radialBasis)

    def _evaluateBasisOnGrid(self, cutoffRadius, res, cropBases, cutoffRadiusInner=None, basis=None):
        samplePointsLinear = torch.linspace(start=-cutoffRadius, end=cutoffRadius, steps=res)
        samplePointsCubic = torch.cartesian_prod(
            samplePointsLinear, samplePointsLinear, samplePointsLinear).view(res, res, res, -1)
        radii = torch.norm(samplePointsCubic, dim=-1)

        if basis is not None:
            samples = basis(samplePointsCubic, radii)
        else:
            samples = self._evaluateJointBasis(samplePointsCubic, radii)

        if cropBases:
            samples[radii > cutoffRadius, :] = 0
            if cutoffRadiusInner is not None: samples[radii < cutoffRadiusInner, :] = 0

        return (samplePointsLinear, samples)

    _basisGridCache = {}

    def _getBasisOnGrid(self, cutoffRadius, res, cropBases, cutoffRadiusInner=None, useCache=True):
        if not useCache:
            return self._evaluateBasisOnGrid(cutoffRadius, res, cropBases, cutoffRadiusInner)

        key = (cutoffRadius, res, cropBases, cutoffRadiusInner)
        if key in self._basisGridCache:
            return self._basisGridCache[key]
        else:
            ret = self._evaluateBasisOnGrid(cutoffRadius, res, cropBases, cutoffRadiusInner)
            self._basisGridCache[key] = ret
            return ret

    def with_peaks_at(self, vectors, values=None):
        r"""Create a spherical tensor with peaks
        The peaks are located in :math:`\vec r_i` and have amplitude :math:`\|\vec r_i \|`
        Parameters
        ----------
        vectors : `torch.Tensor` :math:`\vec r_i` tensor of shape ``(N, 3)``
        values : `torch.Tensor`, optional value on the peak, tensor of shape ``(N)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(self.dim,)``
        """
        if values is None: values = torch.ones(vectors.shape[:-1], dtype=vectors.dtype, device=vectors.device)

        bases = self._evaluateJointBasis(vectors)
        basesSelfDotsInv = 1.0 / torch.einsum('...a,...a->...', bases, bases)
        coeffs = torch.einsum('...b,...,...->b', bases, basesSelfDotsInv, values)
        return coeffs

        # A = torch.einsum(
        #     "ai,bi->ab",
        #     coeff,
        #     coeff
        # )
        # # Y(v_a) . Y(v_b) solution_b = radii_a
        # solution = torch.linalg.lstsq(A, values).solution  #torch.lstsq(values, A).solution.reshape(-1)  # [b]
        # assert (values - A @ solution).abs().max() < 1e-5 * values.abs().max()

        # return solution @ coeff

    def _evaluateSignal(self, signals, basis):
        r"""Expand signal into a weighted sum of bases
        Parameters
        ----------
        signals : `torch.Tensor` of shape ``({... signals}, self.dim)``
        basis : `torch.Tensor` of shape ``({... points}, self.dim)``
        Returns
        -------
        `torch.Tensor` of shape ``({... signals}, {... points})``
        """
        shapeFinal = tuple(list(signals.shape[:-1]) + list(basis.shape[:-1]))

        signalsFlat = signals.view(-1, self.dim)
        basisFlat = basis.view(-1, self.dim)

        retFlat = torch.einsum('sd,pd->sp', signalsFlat, basisFlat)
        ret = retFlat.view(shapeFinal)

        return ret

    def signal_xyz(self, signals, vectors):
        basisValues = self._evaluateJointBasis(vectors)
        return self._evaluateSignal(signals, basisValues)

    def signal_on_grid(self, signals, rcut, res, cropBases=True, cutoffRadiusInner=None, useCache=True):
        samplePointsLinear, samplesBasis = self._getBasisOnGrid(rcut, res, cropBases, cutoffRadiusInner, useCache=useCache)
        return samplePointsLinear, self._evaluateSignal(signals, samplesBasis)


if __name__ == "__main__":
    from e3nn_invstutorial.orthonormal_radial_basis import (
        OrthonormalRadialFunctions,
        FixedCosineRadialModel,
        CosineFunctions,
        FadeAtCutoff
    )


    def orthogonalityTest(basisSamples, integrationWeights):
        nBases = basisSamples.shape[-1]
        orthogonalityCheck = torch.einsum('...b,...c,...->bc', basisSamples, basisSamples, integrationWeights)
        orthonormalityError = torch.max(torch.abs(orthogonalityCheck - torch.eye(nBases)))
        orthogonalityCheck2 = orthogonalityCheck / torch.mean(torch.diag(orthogonalityCheck))
        orthogonalityError = torch.max(torch.abs(orthogonalityCheck2 - torch.eye(nBases)))
        return (orthonormalityError, orthogonalityError, orthogonalityCheck)


    radialCutoff = 3.5
    lmax = 8
    p_val = 1
    p_arg = 1
    nRadialFunctions = (lmax + 1)

    fixedCosineRadialModel = FixedCosineRadialModel(radialCutoff, nRadialFunctions)
    cosineModel = CosineFunctions(nRadialFunctions, radialCutoff)
    cosineModelFaded = FadeAtCutoff(cosineModel, radialCutoff)
    onRadialModel = OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, radialCutoff, 1024)

    rst = RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    # vectors = torch.rand(4,3) - 0.5
    # vectors *= 3.0 / torch.max(torch.norm(vectors,dim=-1))
    # values = torch.rand(vectors.shape[0])
    # signal = rst.with_peaks_at(vectors,values)

    signals = torch.rand((3, rst.dim), dtype=torch.float32)
    points = torch.rand((7, 3), dtype=torch.float32)
    evals = rst.signal_xyz(signals, points)

    # samplePointsLinear, samplesBasis = rst._evaluateBasisOnGrid(radialCutoff, 100, True, None)
    # signalRealSpace = torch.einsum('...b,b->...',samplesBasis,signal)

    # integrationWeightsLinear = torch.ones_like(samplePointsLinear)
    # integrationWeightsLinear[0] = 0.5
    # integrationWeightsLinear[-1] = 0.5
    # integrationWeightsLinear = integrationWeightsLinear / torch.sum(integrationWeightsLinear) * 2*radialCutoff
    # integrationWeightsCubic = torch.einsum('a,b,c->abc',integrationWeightsLinear,integrationWeightsLinear,integrationWeightsLinear)

    # orthonormalityError, orthogonalityError, orthogonalityCheck = orthogonalityTest(samplesBasis, integrationWeightsCubic)
    # print(orthonormalityError, orthogonalityError)
    # np.savetxt('/mnt/c/Users/tjhardi/Documents/overlapMatrixJoint.txt', orthogonalityCheck.numpy())

    print('done')

    # radialCutoff = 3.5
    # nRadialFunctions = 3

    # def radialBasisOriginalConstant(r):
    #     ret = torch.ones((len(r),nRadialFunctions))
    #     for i in range(nRadialFunctions-1): ret[:,i+1] = ret[:,i] * r
    #     return ret

    # onRadialModel = orthonormalRadialBasis.OrthonormalRadialFunctions(nRadialFunctions, radialBasisOriginalConstant, radialCutoff, 1024)
    # np.savetxt('/mnt/c/temp/radialFunctions3.txt', onRadialModel.fSamples.numpy())

    # lmax = 1
    # p_val = 1
    # p_arg = -1
    # rst = RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    # linearSamples, basisGrid = rst._evaluateBasisOnGrid(radialCutoff, 200, True, None)
    # sampleShape = basisGrid.shape[:-1]

    # integrationWeightsLinear = torch.ones_like(linearSamples)
    # integrationWeightsLinear[0] = 0.5
    # integrationWeightsLinear[-1] = 0.5
    # integrationWeightsLinear = integrationWeightsLinear / torch.sum(integrationWeightsLinear) * 2*radialCutoff
    # integrationWeightsCubic = torch.einsum('a,b,c->abc',integrationWeightsLinear,integrationWeightsLinear,integrationWeightsLinear)

    # orthonormalityError, orthogonalityError, orthogonalityCheck = orthogonalityTest(basisGrid, integrationWeightsCubic)
    # print(orthonormalityError, orthogonalityError)
    # np.savetxt('/mnt/c/Users/tjhardi/Documents/overlapMatrix3.txt', orthogonalityCheck.numpy())

    # fixedCosineRadialModel = orthonormalRadialBasis.FixedCosineRadialModel(radialCutoff, nRadialFunctions)
    # cosineModel = orthonormalRadialBasis.CosineFunctions(nRadialFunctions, radialCutoff)
    # cosineModelFaded = orthonormalRadialBasis.FadeAtCutoff(cosineModel, radialCutoff)

    # r = torch.linspace(0,radialCutoff,15)
    # y1 = fixedCosineRadialModel(r)
    # y2 = cosineModel(r)
    # y3 = cosineModelFaded(r)

    # onRadialModel = orthonormalRadialBasis.OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, radialCutoff, 100)

    # lmax = 3
    # p_val = 1
    # p_arg = -1
    # rst = RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    # #positions = torch.tensor([[1.0, 0.0, 0.0],[3.0, 4.0, 0.0]])
    # positions = torch.rand((2,4,6,3))

    # basisPoints = rst._evaluateBasis(positions)

    # samples, basisGrid = rst._evaluateBasisOnGrid(radialCutoff, 75, True, None)

    # basisOverlapMatrix = torch.einsum('xyza,xyzb->ab',basisGrid,basisGrid) * (samples[1]-samples[0])**3
    # print(basisOverlapMatrix)
    # orthogonalityError = torch.max(torch.abs(basisOverlapMatrix - torch.eye(basisOverlapMatrix.shape[0]))) #Should be small relative to 1
    # print(orthogonalityError)

    # np.savetxt('/mnt/c/Users/tjhardi/Documents/overlapMatrix.txt', basisOverlapMatrix.numpy())

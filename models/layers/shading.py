# models/layers/shading.py
import torch
import torch.nn.functional as F

# 经典SH基（Ramamoorthi & Hanrahan），按 n=(nx,ny,nz)
def sph_harm_basis(n):  # n: Bx3xHxW
    nx, ny, nz = n[:,0], n[:,1], n[:,2]
    Y = []
    # l=0
    Y.append(0.282095 * torch.ones_like(nx))
    # l=1
    Y.append(0.488603 * ny)
    Y.append(0.488603 * nz)
    Y.append(0.488603 * nx)
    # l=2
    Y.append(1.092548 * nx * ny)
    Y.append(1.092548 * ny * nz)
    Y.append(0.315392 * (3*nz*nz - 1))
    Y.append(1.092548 * nx * nz)
    Y.append(0.546274 * (nx*nx - ny*ny))
    Y = torch.stack(Y, dim=1)  # Bx9xHxW
    return Y

def shading_from_normals(normals, sh_rgb):  # normals: Bx3xHxW, sh_rgb: Bx27
    B, _, H, W = normals.shape
    Y = sph_harm_basis(normals)  # Bx9xHxW
    sh = sh_rgb.view(B, 3, 9)    # Bx3x9
    # S[c,h,w] = sum_k sh[c,k]*Y[k,h,w]
    S = torch.einsum('bcn,bnhw->bchw', sh, Y)  # Bx3xHxW
    S = torch.clamp(S, min=0.0)  # 阴影非负
    return S

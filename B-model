import numpy as np
import matplotlib.pyplot as plt

# Parameters
B_surface = 1e8           # Tesla, surface field
R_ns = 1e4                # m, neutron star radius
P = 0.1                   # seconds, rotation period (เช่น pulsar)
Omega = 2 * np.pi / P     # angular speed (rad/s)
c = 3e8                   # m/s, speed of light

# Dipole moment
M = B_surface * R_ns**3 / 2

# Grid in x-z plane (y=0)
x = np.linspace(-3*R_ns, 3*R_ns, 201)
z = np.linspace(-3*R_ns, 3*R_ns, 201)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)

# Spherical coordinates
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(z), where=r!=0))
    phi = np.arctan2(y, x)
    return r, theta, phi

r, theta, phi = cartesian_to_spherical(X, Y, Z)

# Dipole B field components
Br = 2 * M * np.cos(theta) / r**3
Btheta = M * np.sin(theta) / r**3
Bphi = np.zeros_like(Br)

# Spherical to Cartesian
Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi)
By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi)
Bz = Br * np.cos(theta) - Btheta * np.sin(theta)

# Goldreich-Julian charge density
rho_gj = -Omega * Bz / (2 * np.pi * c)

# Mask points inside the NS
mask = r < R_ns
rho_gj = np.ma.masked_where(mask, rho_gj)

# Plot
plt.figure(figsize=(7,6))
levels = np.linspace(-np.max(np.abs(rho_gj)), np.max(np.abs(rho_gj)), 31)
cs = plt.contourf(X/R_ns, Z/R_ns, rho_gj, levels=levels, cmap='RdBu_r')
plt.colorbar(cs, label=r'Goldreich-Julian $\rho_{GJ}$ (C/m³)')
plt.xlabel('$x/R_{ns}$')
plt.ylabel('$z/R_{ns}$')
plt.title('GJ Charge Density Distribution in $x$-$z$ Plane')
circle = plt.Circle((0,0), 1, color='k', fill=False)
plt.gca().add_patch(circle)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

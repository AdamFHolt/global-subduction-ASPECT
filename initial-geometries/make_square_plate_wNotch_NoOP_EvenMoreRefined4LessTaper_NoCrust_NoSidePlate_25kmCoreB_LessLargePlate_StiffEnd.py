#!/usr/bin/env python 
import numpy as np
import os, subprocess
import scipy
import scipy.special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.ioff()

# box dimensions 
rmin=3480.e3;rmax=6371.e3;   # distance from center
pmin=0;pmax=2.*np.pi;        # azimuth
tmin=0;tmax=1.*np.pi;        # angle from north

# number of cells
rnum=90
pnum=360
tnum=360 

lith_thick  = 100.e3
notch_radius = 250.e3
notch_depth  = 150.e3
core_thick = 25.e3
Tmax = 1573.0

No_nodes= (rnum + 1) * (pnum + 1) * (tnum + 1) 
C=np.zeros([No_nodes,5],float)
T=np.zeros([No_nodes,4],float)
C_equator=np.zeros([ (rnum + 1) * (pnum + 1)  ,4],float)

# refinement in r-direction
depth_refine = 250.e3    # depth of refinement boundary
num_refine = 80  # number of grid points in refined (upper) layer
lower_lowres =  np.linspace(rmin,rmax-depth_refine,rnum+1-num_refine)
upper_highres = np.linspace(rmax-depth_refine,rmax,1+num_refine)
rvals = np.concatenate((lower_lowres, upper_highres[1:]), axis=0)
print("lower vertical res = %.2f km" % ((rvals[1]-rvals[0])/1.e3))
print("higher vertical res = %.2f km" % ((rvals[rnum]-rvals[rnum-1])/1.e3))

name="comp_square_plate_wNotch_NoOP_EvenMoreRefined4LessTaper_NoCrust_NoSidePlate_25kmCoreB_LessLargePlate_StiffEnd"
f= open(''.join(['text_files/',name,'.txt']),"w+")
f.write("# POINTS: %s %s %s\n" % (str(rnum+1),str(pnum+1),str(tnum+1)))
f.write("# Columns: r phi theta composition1 composition2\n")

name2="temp_square_plate_wNotch_NoOP_EvenMoreRefined4LessTaper_NoCrust_NoSidePlate_25kmCoreB_LessLargePlate_StiffEnd"
f2= open(''.join(['text_files/',name2,'.txt']),"w+")
f2.write("# POINTS: %s %s %s\n" % (str(rnum+1),str(pnum+1),str(tnum+1)))
f2.write("# Columns: r phi theta temperature\n")

width = 8000.e3
length = 8000.e3
width_omega = width * (1./rmax)
length_omega = length * (1./rmax)
shear_zone_width = 100.e3 * (1./rmax)
notch_omega  = notch_radius * (1./rmax)
ridge_length = 1500.e3 * (1./rmax)
ridge_dist = 250.e3 * (1./rmax)
taper_length = 100.e3 * (1./rmax)
stiff_edge_length = ((length/3.)-100.e3) * (1./rmax)

# refinement in theta-direction
num_t_refine = 60  # number of grid points in refined (upper) layer
tres1  =  np.linspace(tmin,0.5*tmax - 0.54*width_omega,((tnum-num_t_refine)/6)+1)
tres2  =  np.linspace(0.5*tmax - 0.54*width_omega,0.5*tmax - 0.46*width_omega,(num_t_refine/2)+1)
tres3  =  np.linspace(0.5*tmax - 0.46*width_omega,0.5*tmax + 0.46*width_omega,(4*(tnum-num_t_refine)/6)+1)
tres4  =  np.linspace(0.5*tmax + 0.46*width_omega,0.5*tmax + 0.54*width_omega,(num_t_refine/2)+1)
tres5  =  np.linspace(0.5*tmax + 0.54*width_omega,tmax,((tnum-num_t_refine)/6)+1)
tvals = np.concatenate((tres1, tres2[1:], tres3[1:],tres4[1:],tres5[1:]), axis=0)
print("t1 trench-parallel res = %.2f km" % ((tvals[1]-tvals[0])*rmax*1.e-3))
print("t2 trench-parallel res = res = %.2f km" % ((tvals[np.size(tres1)+4]-tvals[np.size(tres1)+3])*rmax*1.e-3))

# refinment in phi direction
num_p_refine = 120
pres1 = np.linspace(pmin,1.2*notch_omega,(3*(num_p_refine/6))+1)
pres2 = np.linspace(1.2*notch_omega, pmax - (1.05*length_omega),(4*(pnum-num_p_refine)/6)+1)
pres3 = np.linspace(pmax - (1.05*length_omega),pmax - (0.95*length_omega),(2*(num_p_refine/6))+1)
pres4 = np.linspace(pmax - (0.95*length_omega),pmax-(0.4*notch_omega),(2*(pnum-num_p_refine)/6)+1)
pres5 = np.linspace(pmax-(0.4*notch_omega),pmax,(num_p_refine/6)+1)
pvals = np.concatenate((pres1,pres2[1:],pres3[1:],pres4[1:],pres5[1:]), axis=0)
print("higher  trench-perp res = %.2f km" % ((pvals[1]-pvals[0])*rmax*1.e-3))
print("lower  trench-perp res = %.2f km" % ((pres2[1]-pres2[0])*rmax*1.e-3))
print("higher2 trench-perp res = %.2f km" % ((pres3[1]-pres3[0])*rmax*1.e-3))
print("lower2 trench-perp res = %.2f km" % ((pres4[1]-pres4[0])*rmax*1.e-3))
print("higher3 trench-perp res = %.2f km" % ((pres5[1]-pres5[0])*rmax*1.e-3))

ind=0
print("writting file...")
for t in range(tnum + 1):
	for p in range(pnum + 1):
			for r in range(rnum + 1):

				rad =   rvals[r]
				phi =   pvals[p] 
				theta = tvals[t]
		
				C[ind,0] = rad
				C[ind,1] = phi
				C[ind,2] = theta
			
				T[ind,0] = rad
				T[ind,1] = phi
				T[ind,2] = theta
				T[ind,3] = Tmax

				# flat subduction plate
				if theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi >= (pmax - length_omega) and phi < (pmax - length_omega + taper_length):
					lith_thick_tapered = ((phi - (pmax - length_omega))/taper_length) * (lith_thick)
					if rad >= (rmax - lith_thick_tapered):
						C[ind,4]=1

				elif theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi >= (pmax - length_omega + taper_length) and phi < (pmax - length_omega + stiff_edge_length):
					if rad >= (rmax - lith_thick): 
						C[ind,4]=1

				elif theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi >= (pmax - length_omega + stiff_edge_length): 

					if rad >= (rmax - lith_thick) and rad < (rmax - 0.5*(lith_thick-core_thick) - core_thick):
						C[ind,3]=1
					elif rad >= (rmax  - 0.5*(lith_thick-core_thick) - core_thick) and rad < (rmax - 0.5*(lith_thick-core_thick)): 
						C[ind,4]=1
					elif rad >= (rmax - 0.5*(lith_thick-core_thick)):
						C[ind,3]=1

				# notch
				elif theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi <= (notch_omega):
					z1 = notch_radius;
					z  = rmax - rad
					dx = phi * rmax

					# crust
					if z < (notch_depth):
						dr = ((dx)**2 + (z-z1)**2)
						if dr <= (notch_radius)**2 and dr >= (notch_radius-0.5*(lith_thick-core_thick))**2:
							C[ind,3]=1
						elif dr < (notch_radius-0.5*(lith_thick-core_thick))**2 and dr >= (notch_radius - 0.5*(lith_thick-core_thick) - core_thick)**2:
							C[ind,4]=1
						elif dr < (notch_radius - 0.5*(lith_thick-core_thick) - core_thick)**2 and dr >= (notch_radius - lith_thick)**2:
							C[ind,3]=1


				f.write("%.0f %.4f %.4f %.0f %.0f\n" % (C[ind,0],C[ind,1],C[ind,2],C[ind,3],C[ind,4]))
				f2.write("%.0f %.4f %.4f %.0f\n" % (T[ind,0],T[ind,1],T[ind,2],T[ind,3]))
				ind=ind+1;

f.close()
f2.close()
print("file written.")

ind=0
theta = np.pi/2;
for p in range(pnum + 1):
	for r in range(rnum + 1):

			rad =   rvals[r]
			phi =   pvals[p]

			C_equator[ind,0] = rad
			C_equator[ind,1] = phi


			# flat subduction plate
			if theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi >= (pmax - length_omega) and phi < (pmax - length_omega + taper_length):
				lith_thick_tapered = ((phi - (pmax - length_omega))/taper_length) * (lith_thick)
				if rad >= (rmax - lith_thick_tapered):
					C_equator[ind,3]=1

			elif theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi >= (pmax - length_omega + taper_length) and phi < (pmax - length_omega + stiff_edge_length):
				if rad >= (rmax - lith_thick): 
					C_equator[ind,3]=1

			elif theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi >= (pmax - length_omega + stiff_edge_length): 

				if rad >= (rmax - lith_thick) and rad < (rmax - 0.5*(lith_thick-core_thick) - core_thick):
					C_equator[ind,2]=1
				elif rad >= (rmax  - 0.5*(lith_thick-core_thick) - core_thick) and rad < (rmax - 0.5*(lith_thick-core_thick)): 
					C_equator[ind,3]=1
				elif rad >= (rmax - 0.5*(lith_thick-core_thick)):
					C_equator[ind,2]=1

			elif theta >= (0.5*np.pi - 0.5*width_omega) and theta <= (0.5*np.pi + 0.5*width_omega) and phi <= (notch_omega):
				z1 = notch_radius;
				z  = rmax - rad
				dx = phi * rmax

				# crust
				if z < (notch_depth):
					dr = ((dx)**2 + (z-z1)**2)
					if dr <= (notch_radius)**2 and dr >= (notch_radius-0.5*(lith_thick-core_thick))**2:
						C_equator[ind,2]=1
					if dr < (notch_radius-0.5*(lith_thick-core_thick))**2 and dr >= (notch_radius - 0.5*(lith_thick-core_thick) - core_thick)**2:
						C_equator[ind,3]=1
					if dr < (notch_radius - 0.5*(lith_thick-core_thick) - core_thick)**2 and dr >= (notch_radius - lith_thick)**2:
						C_equator[ind,2]=1


			ind=ind+1;


f.close()

# plot to check
print("plotting...")
plot_name_pdf = ''.join(['test_plots/',str(name),'.1.pdf'])
plot_name_png = ''.join(['test_plots/',str(name),'.1.png'])

fig, ax = plt.subplots()
x = C_equator[:,0] * np.sin(C_equator[:,1])
y = C_equator[:,0] * np.cos(C_equator[:,1])
comps = plt.scatter(x,y,c=C_equator[:,2],cmap='bwr',vmin=0,vmax=1,s=0.25,lw=0)
plt.xlim(-1.0*rmax-100.e3,rmax+100.e3)
plt.ylim(-1.0*rmax-100.e3,rmax+100.e3)
# shade the core
fill_array_x = np.zeros((50))
fill_array_y = np.zeros((50))
for i in range(len(fill_array_x)):
		theta = i * ((2*np.pi)/(len(fill_array_x)-1.))
		fill_array_x[i] = rmin * np.sin(theta)
		fill_array_y[i] = rmin * np.cos(theta)
plt.fill(fill_array_x, fill_array_y, "gray");
cb = plt.colorbar(comps)

bash_command = ''.join(['convert -density 400 -flatten ',plot_name_pdf,' ',plot_name_png]);
plt.savefig(plot_name_pdf, bbox_inches='tight', format='pdf')
process = subprocess.Popen(['/bin/bash','-c',bash_command])
process.wait()
os.remove(plot_name_pdf)
print("plot saved at %s" % plot_name_png)

print("plotting 2nd...")
plot_name_pdf = ''.join(['test_plots/',str(name),'.2.pdf'])
plot_name_png = ''.join(['test_plots/',str(name),'.2.png'])

fig, ax = plt.subplots()
x = C_equator[:,0] * np.sin(C_equator[:,1])
y = C_equator[:,0] * np.cos(C_equator[:,1])
comps = plt.scatter(x,y,c=C_equator[:,3],cmap='bwr',vmin=0,vmax=1,s=0.25,lw=0)
plt.xlim(-1.0*rmax-100.e3,rmax+100.e3)
plt.ylim(-1.0*rmax-100.e3,rmax+100.e3)
# shade the core
fill_array_x = np.zeros((50))
fill_array_y = np.zeros((50))
for i in range(len(fill_array_x)):
		theta = i * ((2*np.pi)/(len(fill_array_x)-1.))
		fill_array_x[i] = rmin * np.sin(theta)
		fill_array_y[i] = rmin * np.cos(theta)
plt.fill(fill_array_x, fill_array_y, "gray");
cb = plt.colorbar(comps)

bash_command = ''.join(['convert -density 400 -flatten ',plot_name_pdf,' ',plot_name_png]);
plt.savefig(plot_name_pdf, bbox_inches='tight', format='pdf')
process = subprocess.Popen(['/bin/bash','-c',bash_command])
process.wait()
os.remove(plot_name_pdf)
print("plot saved at %s" % plot_name_png)


bash_command = ''.join(['convert -density 400 -flatten ',plot_name_pdf,' ',plot_name_png]);
plt.savefig(plot_name_pdf, bbox_inches='tight', format='pdf')
process = subprocess.Popen(['/bin/bash','-c',bash_command])
process.wait()
os.remove(plot_name_pdf)
print("plot saved at %s" % plot_name_png)



from PyHEADTAIL.general.element import Element
import numpy as np
import h5py as hp 
from Visualisations import plot_particle_loss

class ParticleLoss(Element):

	def __init__(self, percent, macroparticlenumber, path, axis, Qp):
		self.critical_n = macroparticlenumber*percent/100
		self.y_max = 10e-3
		self.x_max = 25e-3
		self.dict_ = dict()
		self.path = path 
		self.axis = axis
		self.Qp = Qp
	
	def update(self, current, turn):
		self.current = current
		self.turn = turn 
		self.filename = f'Qp={self.Qp}_current={current:.3}_turn={turn}_{self.axis}'
		
	def track(self, bunch):
		bunch_r = bunch.y**2/self.y_max**2+bunch.x**2/self.x_max**2
		bool_i = np.nonzero((bunch_r >= 1).get())[0]

		if len(bool_i) >= self.critical_n:
			bunch_x = bunch.x.copy()
			bunch_x = bunch_x.get()
			bunch_x = bunch_x[bool_i]
			self.dict_['x'] = bunch_x
	
			bunch_y = bunch.y.copy()
			bunch_y = bunch_y.get()
			bunch_y = bunch_y[bool_i]
			self.dict_['y'] = bunch_y	
			
			plot_particle_loss(self.dict_, name=self.filename, path=self.path)	
			with open(self.path+'current_transmition.txt','a') as f:
				f.write(f'{self.Qp}\t{self.current:.3}\n')
			#	f.create_dataset("default", data=self.dict_)
			raise KeyboardInterrupt
			
			

		
		

# import plot functions
from Animate_SWE import Animate_PDE_Losses
from Plots_SWE import (
    Plot_Boundary_Conditions,
    Plot_Initial_Conditions,
    Plot_Learning_Curve,
    Plot_PDE_Losses,
    Plot_Results,
    Plot_Solution_Over_Training,
    Plot_PDE_Loss_Over_Training,
)

# plot results from file
Plot_Learning_Curve()
Plot_Results()
Plot_Initial_Conditions()
Plot_Boundary_Conditions()
Plot_PDE_Losses()
Plot_Solution_Over_Training(epochs=10000)
Plot_PDE_Loss_Over_Training(epochs=10000)

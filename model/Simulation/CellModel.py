from cc3d import CompuCellSetup

from CellModelSteppables import CellModelSteppable

CompuCellSetup.register_steppable(steppable=CellModelSteppable(frequency=1))

CompuCellSetup.run()

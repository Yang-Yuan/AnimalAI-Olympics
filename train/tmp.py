# from pynput import keyboard
#
# def on_press(key):
#     try:
#         print('alphanumeric key {0} pressed'.format(
#             key.char))
#     except AttributeError:
#         print('special key {0} pressed'.format(
#             key))
#
# def on_release(key):
#     print('{0} released'.format(
#         key))
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False
#
# # Collect events until released
# with keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()
#################################################################################


# from statemachine import StateMachine, State
#
#
# class CampaignMachineWithKeys(StateMachine):
#     "A workflow machine"
#     draft = State('Draft', initial=True, value=1)
#     producing = State('Being produced', value=2)
#     closed = State('Closed', value=3)
#
#     add_job = draft.to.itself() | producing.to.itself()
#     produce = draft.to(producing)
#     deliver = producing.to(closed)
#
#
# class MyModel(CampaignMachineWithKeys):
#     state_machine_name = 'CampaignMachineWithKeys'
#
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#         super(MyModel, self).__init__()
#
#     def __repr__(self):
#         return "{}({!r})".format(type(self).__name__, self.__dict__)
#
#
# model = MyModel(state='draft')
# assert isinstance(model.statemachine, campaign_machine)
# assert model.state == 'draft'
# assert model.statemachine.current_state == model.statemachine.draft

# from pathfinding.core.diagonal_movement import DiagonalMovement
# from pathfinding.core.grid import Grid
# from pathfinding.finder.a_star import AStarFinder
#
# matrix = [
#   [1, 1, 1],
#   [1, 0, 1],
#   [1, 1, 1]
# ]
# grid = Grid(matrix=matrix)
#
# start = grid.node(0, 0)
# end = grid.node(99, 99)
#
# finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
# path, runs = finder.find_path(start, end, grid)
#
# print('operations:', runs, 'path length:', len(path))
# print(grid.grid_str(path=path, start=start, end=end))


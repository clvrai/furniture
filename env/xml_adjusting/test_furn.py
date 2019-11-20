from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys

model_path = sys.argv[1]
model = load_model_from_path(model_path)
sim = MjSim(model)
mjmodel = sim.model


body_id2name = mjmodel.body_id2name
# for i, (id1, id2) in enumerate(zip(mjmodel.eq_obj1id, mjmodel.eq_obj2id)):
# 	print("weld joint", i, "body1=" + body_id2name(id1) + ", body2=" + body_id2name(id2) + ", active=" + str(mjmodel.eq_active[i]))

# print(mjmodel.eq_active)
# mjmodel.eq_active[0] = 0
# print(mjmodel.eq_active[0])

viewer = MjViewer(sim)
for i in range(15000):
    # if (i+1) %1000 == 0:
    #     print("here")
    #     mjmodel.eq_active[0] = 0
        # mjmodel.eq_active[1] = 0
        # mjmodel.eq_active[2] = 0

    sim.step()
    viewer.render()




                   # #print(dir(env))
                   #  print("eq", env.sim.model.eq_active)
                   #  body_id2name = env.sim.model.body_id2name
                   #  print("here", env.sim.model.eq_obj1id, env.sim.model.eq_obj2id)
                   #  for i, (id1, id2) in enumerate(zip(env.sim.model.eq_obj1id, env.sim.model.eq_obj2id)):
                   #      print("weld joint", i, "body1=" + body_id2name(id1) + ", body2=" + body_id2name(id2) + ", active=" + str(mjmodel.eq_active[i]))

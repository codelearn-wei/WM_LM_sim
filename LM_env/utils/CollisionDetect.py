# 粗略的碰撞检测(视作圆形)  如果此时不碰撞，就无需按矩形检测。返回的距离作为该点车到障碍物的大致距离
import math
def ColliTestRough(ego, obs,egoLength, egoWidth)->bool:

    dis = math.sqrt((ego.x - obs.x) ** 2 + (ego.y - obs.y) ** 2)

    max_veh = max(egoLength, egoWidth)
    max_obs = max(obs.length*1.5, obs.width*2)
    return dis - (max_veh + max_obs) / 2 <=0

def ColliTestPrecise(ego, obs,egoLength, egoWidth) -> bool:
    """检测某个步长ego和obstalce是否发生碰撞

    Args:
        ego (_type_): ego的trajego类
        obs (_type_): obstacle的类

    Returns:
        True --> collide
    """
    shift_x = obs.x - ego.x
    shift_y = obs.y - ego.y

    # global VEH_L, VEH_W
    cos_v = math.cos(ego.yaw)
    sin_v = math.sin(ego.yaw)  # （cos，sin）是ego的朝向，单位向量
    cos_o = math.cos(obs.heading)
    sin_o = math.sin(obs.heading)
    half_l_v = egoLength / 2
    half_w_v = egoWidth / 2
    half_l_o = obs.length / 2*2
    half_w_o = obs.width / 2*1.5

    dx1 = cos_v * egoLength / 2
    dy1 = sin_v * egoLength / 2
    dx2 = sin_v * egoWidth/ 2
    dy2 = -cos_v * egoWidth / 2
    dx3 = cos_o * obs.length / 2
    dy3 = sin_o * obs.length / 2
    dx4 = sin_o * obs.width / 2
    dy4 = -cos_o * obs.width / 2

    # 使用分离轴定理进行碰撞检测
    # a·b a在b上的投影
    return ((abs(shift_x * cos_v + shift_y * sin_v) <=
             abs(dx3 * cos_v + dy3 * sin_v) + abs(dx4 * cos_v + dy4 * sin_v) + half_l_v)
            and (abs(shift_x * sin_v - shift_y * cos_v) <=
                 abs(dx3 * sin_v - dy3 * cos_v) + abs(dx4 * sin_v - dy4 * cos_v) + half_w_v)
            and (abs(shift_x * cos_o + shift_y * sin_o) <=
                 abs(dx1 * cos_o + dy1 * sin_o) + abs(dx2 * cos_o + dy2 * sin_o) + half_l_o)
            and (abs(shift_x * sin_o - shift_y * cos_o) <=
                 abs(dx1 * sin_o - dy1 * cos_o) + abs(dx2 * sin_o - dy2 * cos_o) + half_w_o))
    

def ColliTest(ego,obs,egoLength,egoWidth) -> bool:
    
    if ColliTestRough(ego,obs,egoLength,egoWidth): #粗略检测
        if ColliTestPrecise(ego,obs,egoLength,egoWidth): #细致检测
            return True
    
    return False 
    



   

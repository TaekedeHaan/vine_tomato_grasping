#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from state_machine.state_machine_input import StateMachineInput
from state_machine.model_spawner_state_machine import ModelSpawnerStateMachine
from spawn_truss import ModelSpawner

NODE_NAME = 'model_spawner'
DEFAULT_UPDATE_RATE = 20.0
DEFAULT_DEBUG_MODE = True

def main():
    debug_mode = rospy.get_param(NODE_NAME + "/debug", DEFAULT_DEBUG_MODE)
    if debug_mode:
        log_level = rospy.DEBUG
    else:
        log_level = rospy.INFO

    rospy.init_node(NODE_NAME, log_level=log_level)

    update_rate = rospy.get_param('~update_rate', DEFAULT_UPDATE_RATE)

    state_machine_input = StateMachineInput(NODE_NAME)
    model_spawner = ModelSpawner()
    model_spawner_state_machine = ModelSpawnerStateMachine(model_spawner, state_machine_input, update_rate, NODE_NAME)
    rospy.loginfo('[%s] Model spawner state machine successfully generated', NODE_NAME)

    rospy.core.add_preshutdown_hook(lambda reason: model_spawner_state_machine.request_shutdown())

    model_spawner_state_machine.run()


if __name__ == '__main__':
    main()
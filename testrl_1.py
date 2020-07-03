# FIRST SET UP ENVIRONMENT

import sys
import os
from typing import List
import time
sys.path.append(os.path.abspath('../..'))
from utils import common
from utils.topics import Topic
from services.service import Service, PublishSubscribe, RemoteService

from utils.domain.domain import Domain
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger, LogLevel

from services.hci import ConsoleInput, ConsoleOutput
from services.nlu import HandcraftedNLU
from services.bst import HandcraftedBST
from services.ust import HandcraftedUST
from services.policy import HandcraftedPolicy
from services.nlg import HandcraftedNLG
from services.domain_tracker import DomainTracker

from services.service import DialogSystem

from tensorboardX import SummaryWriter
from services.policy.rl.experience_buffer import NaivePrioritizedBuffer
from services.simulator import HandcraftedUserSimulator
from services.policy import DQNPolicy
from services.stats.evaluation import PolicyEvaluator


super_domain = JSONLookupDomain(name="ImsLecturers")

policy = HandcraftedPolicy(domain=super_domain)

# Allows you to track training progress using tensorboard
summary_writer = SummaryWriter(os.path.join('logs', "tutorial"))

# logs summary statistics for each train/test epoch
logger = DiasysLogger(console_log_lvl=LogLevel.RESULTS, file_log_lvl=LogLevel.DIALOGS)
dialogue_logger = DiasysLogger(name='dialogue_logger', 
    console_log_lvl=LogLevel.ERRORS, 
    file_log_lvl=LogLevel.DIALOGS,
    logfile_folder='dialogue_history',
    logfile_basename='history')

# Create RL policy instance with parameters used in ADVISER paper
policy = DQNPolicy(
    domain=super_domain, lr=0.0001, eps_start=0.3, gradient_clipping=5.0,
    buffer_cls=NaivePrioritizedBuffer, replay_buffer_size=8192, shared_layer_sizes=[256],
    train_dialogs=1000, target_update_rate=3, training_frequency=2, logger=dialogue_logger,
    summary_writer=summary_writer)

user_sim = HandcraftedUserSimulator(domain=super_domain, logger=dialogue_logger)

evaluator = PolicyEvaluator(
    domain=super_domain, use_tensorboard=True,
    experiment_name="tutorial",
    logger=dialogue_logger, summary_writer=summary_writer)


# SET CONSTANTS
TRAIN_EPOCHS = 10
TRAIN_EPISODES = 1000
EVAL_EPISODES = 1000
MAX_TURNS = 25
bst = HandcraftedBST(domain=super_domain, logger=dialogue_logger)

#user state
ust = HandcraftedUST(domain=super_domain, logger=dialogue_logger)

# Choose how many repeated trials
for i in range(1):
    common.init_random()  # add seed here as a parameter to the init_random if wanted

    ds = DialogSystem(services=[user_sim, bst, ust, policy, evaluator], protocol='tcp')

    # Start train/eval loop
    for j in range(TRAIN_EPOCHS):
        # START TRAIN EPOCH
        evaluator.train()
        policy.train()
        evaluator.start_epoch()
        for episode in range(TRAIN_EPISODES):
            if episode % 100 == 0:
                print("DIALOG", episode)
            logger.dialog_turn("\n\n!!!!!!!!!!!!!!!! NEW DIALOG !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            ds.run_dialog(start_signals={f'user_acts/{super_domain.get_domain_name()}': []})
        evaluator.end_epoch()

        # START EVAL EPOCH
        evaluator.eval()
        policy.eval()
        evaluator.start_epoch()
        for episode in range(EVAL_EPISODES):
            logger.dialog_turn("\n\n!!!!!!!!!!!!!!!! NEW DIALOG !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            ds.run_dialog(start_signals={f'user_acts/{super_domain.get_domain_name()}': []})
        evaluator.end_epoch()
    policy.save()
    ds.shutdown()

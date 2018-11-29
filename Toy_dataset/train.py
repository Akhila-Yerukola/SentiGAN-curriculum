import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_Data_loader
import pickle
from target_lstm import TARGET_LSTM
from generator import Generator
from discriminator import Discriminator
import argparse
# from rollout import ROLLOUT

parser = argparse.ArgumentParser(description='SentiGAN with curriculum training')
parser.add_argument('--seq_len', type=int, default=1,
                    help='sequence length to start curriculum training from')
parser.add_argument('--max_seq_len', type=int, default=20,
                    help='sequence length to end curriculum training at')
parser.add_argument('--save', type=str, default='save',
                    help='location of save the model and logs')
parser.add_argument('--disc_pre_epoch', type=int, default=4,
                    help='discriminator pre train epochs')
parser.add_argument('--gen_pre_epoch', type=int, default=100,
                    help='discriminator pre train epochs')
parser.add_argument('--adversarial_epoch', type=int, default=350,
                    help='adversarial training epochs')
args = parser.parse_args()

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = args.max_seq_len # sequence length

SEED = 88
BATCH_SIZE = 64


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 50
import os
if not os.path.exists(args.save):
    os.makedirs(args.save) 
positive_file = args.save + '/real_data.txt'
negative_file = args.save + '/generator_sample.txt'
eval_file = args.save + '/eval_file.txt'
generated_num = 10000
vocab_size = 5000
START_TOKEN = 0



def generate_samples_from_target(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Target Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        #import pdb
        #pdb.set_trace()
        _, g_loss = trainable_model.pretrain_step(sess, batch, START_TOKEN)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


# def train_with_good_rewards(sess, data_loader):
#     data_loader.reset_pointer()



def main():
    random.seed(SEED)
    np.random.seed(SEED)
    seq_len = args.seq_len

    # prepare data
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)  # For testing
    dis_data_loader = Dis_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    generator = None
    discriminator = None
    target_lstm = None


    while seq_len <= SEQ_LENGTH:
        log = open(args.save + '/experiment-log' + str(seq_len) + '.txt', 'w')
        print("Current sequence length is " + str(seq_len))
        print('Args:', args)
        log.write(str(args))
        if generator is None:
            log.write("Init generator")
            print("Init generator")
        else:
            log.write("Used same generator")
            print("Used same generator")
        generator = Generator(num_emb=vocab_size, batch_size=BATCH_SIZE, emb_dim=EMB_DIM, 
            num_units=HIDDEN_DIM, sequence_length=SEQ_LENGTH, start_token=START_TOKEN, 
            true_seq_len=seq_len, save_model_path = args.save) if generator is None else generator
        generator.true_seq_len = seq_len

        # target_params's size: [15 * 5000 * 32]
        target_params = pickle.load(open('./save/target_params_py3.pkl', 'rb'))
        # The oracle model
        target_lstm = TARGET_LSTM(5000, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, 0, target_params, seq_len) if target_lstm is None else target_lstm
        target_lstm.true_seq_len = seq_len
        discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                      embedding_size=dis_embedding_dim,
                                      filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                      l2_reg_lambda=dis_l2_reg_lambda, save_model_path=args.save) if discriminator is None else discriminator

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        generate_samples_from_target(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
        gen_data_loader.create_batches(positive_file, seq_len)

        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #
        # likelihood_data_loader.create_batches(positive_file)
        # for i in range(100):
        #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        #     print('my step ', i, 'test_loss ', test_loss)
        #     input("next:")
        # input("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        #log = open('save_19_20/experiment-log' + str(seq_len) + '.txt', 'w')
        #  pre-train generator
        print('Start pre-training...')
        log.write('pre-training...\n')
        ans_file = open(args.save + '/learning_cure' + str(seq_len) + '.txt', 'w')
        epochs = args.gen_pre_epoch 
        #ans_file.write("-------- %s \n" % seq_len)
        for epoch in range(epochs):  # 120
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % 1 == 0:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file, seq_len)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                log.write(buffer)
                ans_file.write("%s\n" % str(test_loss))

        buffer = 'Start pre-training discriminator...'
        print(buffer)
        log.write(buffer)
        for _ in range(args.disc_pre_epoch):   # 10
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file, seq_len)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                    }
                    d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op], feed)
            buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
            print(buffer)

            log.write(buffer)
        ans_file.write("==========\n")
        print("Start Adversarial Training...")
        log.write('adversarial training...')
        TOTAL_BATCH = args.adversarial_epoch
        for total_batch in range(TOTAL_BATCH):
            # Train the generator
            for it in range(1):
                samples = generator.generate(sess)
                rewards = generator.get_reward(sess, samples, 16, discriminator, START_TOKEN)
                a = str(samples[0])
                b = str(rewards[0])
                buffer = "%s\n%s\n\n" % (a, b)
                # print(buffer)
                log.write(buffer)
                rewards_loss, mle_loss = generator.update_with_rewards(sess, samples, rewards, START_TOKEN)

            # Test
            if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file, seq_len)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'reward-train epoch %s train loss %s mle loss %s test_loss %s\n' % (str(total_batch), str(rewards_loss), str(mle_loss), str(test_loss))
                print(buffer)
                log.write(buffer)
                ans_file.write("%s\n" % str(test_loss))
                
            if total_batch % 20 == 0 or total_batch == TOTAL_BATCH - 1: generator.save_model(sess, seq_len)

            # Train the discriminator
            for _ in range(1):
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                dis_data_loader.load_train_data(positive_file, negative_file, seq_len)
                for _ in range(3):
                    dis_data_loader.reset_pointer()
                    for it in range(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                        }
                        d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op], feed)
                if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                    buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
                    print(buffer)
                    log.write(buffer)
                if total_batch %20 == 0 or total_batch == TOTAL_BATCH - 1: discriminator.save_model(sess, seq_len)
        seq_len += 1




if __name__ == '__main__':
    main()




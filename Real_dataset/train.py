import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_Data_loader
import pickle
from generator import Generator
from discriminator import Discriminator
# from rollout import ROLLOUT
import argparse
parser = argparse.ArgumentParser(description='SentiGAN with curriculum training')
parser.add_argument('--seq_len', type=int, default=1,
                    help='sequence length to start curriculum training from')
parser.add_argument('--max_seq_len', type=int, default=17,
                    help='sequence length to end curriculum training at')
parser.add_argument('--save', type=str, default='save',
                    help='location of save the model and logs')
parser.add_argument('--disc_pre_epoch', type=int, default=10,
                    help='discriminator pre train epochs')
parser.add_argument('--gen_pre_epoch', type=int, default=150,
                    help='discriminator pre train epochs')
parser.add_argument('--adversarial_epoch', type=int, default=2000,
                    help='adversarial training epochs')
parser.add_argument('--lbda', type=float, default=1,
                    help='weightage of mle during adv training')
args = parser.parse_args()


#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 200 # embedding dimension
HIDDEN_DIM = 200 # hidden state dimension of lstm cell
MAX_SEQ_LENGTH = args.max_seq_len  # max sequence length
BATCH_SIZE = 64


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = args.adversarial_epoch
import os
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.exists(args.save + '/infer'):
    os.makedirs(args.save + '/infer') 
dataset_path = "data/movie/"
emb_dict_file = dataset_path + "imdb_word.vocab"

# imdb corpus
imdb_file_txt = dataset_path + "imdb/imdb_sentences.txt"
imdb_file_id = dataset_path + "imdb/imdb_sentences.id"

# sstb corpus
sst_pos_file_txt = dataset_path + 'sstb/sst_pos_sentences.txt'
sst_pos_file_id = dataset_path + 'sstb/sst_pos_sentences.id'
sst_neg_file_txt = dataset_path + 'sstb/sst_neg_sentences.txt'
sst_neg_file_id = dataset_path + 'sstb/sst_neg_sentences.id'


eval_file = args.save + '/eval_file.txt'
eval_text_file = args.save + '/eval_text_file.txt'
negative_file = args.save + '/generator_sample.txt'
infer_file = args.save + '/infer/'


def generate_samples(sess, trainable_model, generated_num, output_file, vocab_list, if_log=False, epoch=0):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num)):
        generated_samples.extend(trainable_model.generate(sess))

    if if_log:
        mode = 'a'
        if epoch == 0:
            mode = 'w'
        with open(eval_text_file, mode) as fout:
            # id_str = 'epoch:%d ' % epoch
            for poem in generated_samples:
                poem = list(poem)
                if 2 in poem:
                    poem = poem[:poem.index(2)]
                buffer = ' '.join([vocab_list[x] for x in poem]) + '\n'
                fout.write(buffer)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            poem = list(poem)
            if 2 in poem:
                poem = poem[:poem.index(2)]
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_infer(sess, trainable_model, epoch, vocab_list):
    generated_samples = []
    for _ in range(int(100)):
        # generated_samples.extend(trainable_model.infer(sess))
        generated_samples.extend(trainable_model.generate(sess))
    file = infer_file+str(epoch)+'.txt'
    with open(file, 'w') as fout:
        for poem in generated_samples:
            poem = list(poem)
            if 2 in poem:
                poem = poem[:poem.index(2)]
            buffer = ' '.join([vocab_list[x] for x in poem]) + '\n'
            fout.write(buffer)
    print("%s saves" % file)
    return


def produce_samples(generated_samples):
    produces_sample = []
    for poem in generated_samples:
        poem_list = []
        for ii in poem:
            if ii == 0:  # _PAD
                continue
            if ii == 2:  # _EOS
                break
            poem_list.append(ii)
        produces_sample.append(poem_list)
    return produces_sample


def load_emb_data(emb_dict_file):
    word_dict = {}
    word_list = []
    item = 0
    with open(emb_dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            word_dict[word] = item
            item += 1
            word_list.append(word)
    length = len(word_dict)
    print("Load embedding success! Num: %d" % length)
    return word_dict, length, word_list


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(200):  # data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():

    # load embedding info
    vocab_dict, vocab_size, vocab_list = load_emb_data(emb_dict_file)
    seq_len = args.seq_len
    # prepare data
    pre_train_data_loader = Gen_Data_loader(BATCH_SIZE, vocab_dict, MAX_SEQ_LENGTH)
    

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, vocab_dict, MAX_SEQ_LENGTH)
    

    dis_data_loader = Dis_Data_loader(BATCH_SIZE, vocab_dict, MAX_SEQ_LENGTH)
    generator = None
    discriminator = None
    target_lstm = None
    while seq_len <= MAX_SEQ_LENGTH:
        # build model
        # num_emb, vocab_dict, batch_size, emb_dim, num_units, sequence_length
        pre_train_data_loader.create_batches([imdb_file_id, sst_pos_file_id, sst_neg_file_id], seq_len)
        gen_data_loader.create_batches([sst_pos_file_id, sst_neg_file_id], seq_len)
        generator = Generator(num_emb = vocab_size, vocab_dict = vocab_dict, batch_size = BATCH_SIZE, emb_dim = EMB_DIM, num_units = HIDDEN_DIM,
                 max_sequence_length = MAX_SEQ_LENGTH, true_seq_len=seq_len, save_model_path = args.save, lbda=args.lbda) if generator is None else generator
        discriminator = Discriminator(sequence_length=MAX_SEQ_LENGTH, num_classes=2,
                                      vocab_size=vocab_size,
                                      embedding_size=dis_embedding_dim,
                                      filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                      l2_reg_lambda=dis_l2_reg_lambda, save_model_path=args.save) if discriminator is None else discriminator

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        log = open(args.save + '/experiment-log' + str(seq_len) + '.txt', 'w')

        buffer = 'Start pre-training generator...'
        print(buffer)
        log.write(buffer + '\n')
        for epoch in range(args.gen_pre_epoch):  #120
            train_loss = pre_train_epoch(sess, generator, pre_train_data_loader)
            if epoch % 5 == 0:
                generate_samples(sess, generator, 1, eval_file, vocab_list, if_log=True, epoch=epoch)
                print('    pre-train epoch ', epoch, 'train_loss ', train_loss)
                buffer = '    epoch:\t' + str(epoch) + '\tnll:\t' + str(train_loss) + '\n'
                log.write(buffer)

        buffer = 'Start pre-training discriminator...'
        print(buffer)
        log.write(buffer)
        for _ in range(args.disc_pre_epoch):   # 10
            generate_samples(sess, generator, 70, negative_file, vocab_list)
            dis_data_loader.load_train_data([sst_pos_file_id, sst_neg_file_id], [negative_file])
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
            buffer = "discriminator loss %f acc %f" % (d_loss, d_acc)
            print(buffer)
            log.write(buffer + '\n')

        print("Start Adversarial Training...")
        log.write('adversarial training...')
        for total_batch in range(TOTAL_BATCH):
            # Train the generator
            for it in range(2):
                # print("1")
                samples = generator.generate(sess)
                samples = produce_samples(samples)
                # print("2")
                rewards = generator.get_reward(sess, samples, 16, discriminator)
                # print("3")
                a = str(samples[0])
                b = str(rewards[0])
                # rewards = change_rewards(rewards)
                # c = str(rewards[0])
                d = build_from_ids(samples[0], vocab_list)
                buffer = "%s\n%s\n%s\n\n" % (d, a, b)
                print(buffer)
                log.write(buffer)

                # print("4")
                rewards_loss = generator.update_with_rewards(sess, samples, rewards)
                # print("5")
                
                # little1 good reward
                little1_samples = gen_data_loader.next_batch()
                rewards = generator.get_reward(sess, little1_samples, 16, discriminator)
                a = str(little1_samples[0])
                b = str(rewards[0])
                buffer = "%s\n%s\n\n" % (a, b)
                # print(buffer)
                log.write(buffer)
                rewards_loss = generator.update_with_rewards(sess, little1_samples, rewards)

            # generate_infer(sess, generator, epoch, vocab_list)

            # Test
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                generate_samples(sess, generator, 120, eval_file, vocab_list, if_log=True)
                generate_infer(sess, generator, total_batch, vocab_list)
                buffer = 'reward-train epoch %s train loss %s' % (str(total_batch), str(rewards_loss))
                print(buffer)
                log.write(buffer + '\n')
                

            if total_batch % 20 == 0 or total_batch == TOTAL_BATCH - 1: generator.save_model(sess, seq_len)

            # Train the discriminator
            begin = True
            for _ in range(1):
                generate_samples(sess, generator, 70, negative_file, vocab_list)
                dis_data_loader.load_train_data([sst_pos_file_id, sst_neg_file_id], [negative_file])
                for _ in range(3):
                    dis_data_loader.reset_pointer()
                    for it in range(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                        }
                        d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op],
                                                    feed)
                        if (total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1) and begin:
                            buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
                            print(buffer)
                            log.write(buffer)
                            begin = False
            if total_batch %20 == 0 or total_batch == TOTAL_BATCH - 1: discriminator.save_model(sess, seq_len)
            # pretrain
            for _ in range(10):
                train_loss = pre_train_epoch(sess, generator, pre_train_data_loader)



def build_from_ids(vv, vocab_list):
    a = []
    for i in vv:
        a.append(vocab_list[i])
    return(' '.join(a))


if __name__ == '__main__':
    main()



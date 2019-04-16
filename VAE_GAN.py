#-*- coding: utf-8 -*-
# 部分注释可能有误，大家将就看看，我这几天抽空改一下
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *
from glob import glob

import prior_factory as prior

class VAE_GAN(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        #self.test_dir = test_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.input_fname_pattern = '*.jpg'
        self.model_name = "VAE_GAN"     # name for checkpoint

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.r = 0.05
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError
    # Gaussian Encoder 高斯编码器
    def encoder(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):
            if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
                net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc6')
            # The mean parameter is unconstrained 将输出分为两块，z_mean和z_log_var,就是高斯分布的均值和标准差
            # 分出的前（64,62）代表均值， 后（64,62）处理后代表标准差
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            # 标准差必须是正值。 用softplus参数化并为数值稳定性添加一个小的epsilon, softplus为y=log(1+ex)
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

            return mean, stddev

    def decoder(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
                net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
                net = tf.reshape(net, [self.batch_size, 7, 7, 128])
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
                       scope='de_bn3'))
                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='de_dc4'))
            return out

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
                net = lrelu(
                    bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))  # 数据标准化
                netf = tf.reshape(net, [self.batch_size, -1])
                netf = MinibatchLayer(32, 32, netf, 'd_fc3')
                netf = lrelu(bn(linear(netf, 1024, scope='d_fc4'), is_training=is_training, scope='d_bn4'))
                out_logit = linear(netf, 1, scope='d_fc5')
                out = tf.nn.sigmoid(out_logit)
            return out, out_logit, net

    def NLLNormal(self, pred, target):
        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c
        return tmp

    def build_model(self):
        # some parameters
        image_dims = [self.output_height, self.output_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """
        # encoding
        self.mu, sigma = self.encoder(self.inputs, is_training=True, reuse=False)

        # sampling by re-parameterization technique
        # tf.random_normal 从正态分布输出随机值, sigma乘上随机值后将服从正态分布，抽空博客仔细说一下
        z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        # decoding
        out_dec = self.decoder(z, is_training=True, reuse=False)
        self.out_dec = tf.clip_by_value(out_dec, 1e-8, 1 - 1e-8)

        # VAE loss
        marginal_likelihood_dec = tf.reduce_sum(
            self.inputs * tf.log(self.out_dec) + (1 - self.inputs) * tf.log(1 - self.out_dec), [1, 2])
        # 在我写的Word中有对KL实现的讲解
        KL_divergence = -0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1,
            [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood_dec)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        #ELBO = -self.neg_loglikelihood - self.KL_divergence

        # GAN loss
        D_real, D_real_logits, net_r = self.discriminator(self.inputs, is_training=True, reuse=False)
        out_dec_noise = self.decoder(self.z, is_training=True, reuse=True)
        D_fake_dec_ns, D_fake_logits_ns, _ = self.discriminator(out_dec_noise, is_training=True, reuse=True)
        D_fake_dec, D_fake_logits_dec, net_f = self.discriminator(self.out_dec, is_training=True, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake_dec_ns = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_ns, labels=tf.zeros_like(D_fake_dec_ns)))
        d_loss_fake_dec = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_dec, labels=tf.zeros_like(D_fake_dec)))

        dec_loss_d1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_dec, labels=tf.ones_like(D_fake_dec)))
        dec_loss_d2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_ns, labels=tf.ones_like(D_fake_dec_ns)))
        dec_loss_d = dec_loss_d1 + dec_loss_d2

        self.LL_loss = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(net_f, net_r), [1,2,3]))
        self.d_loss = d_loss_real + d_loss_fake_dec + d_loss_fake_dec_ns
        self.dec_loss = dec_loss_d - 1e-8*self.LL_loss
        self.enc_loss = -self.LL_loss/(4 * 4 * 256) - self.KL_divergence/(self.z_dim*self.batch_size)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        dec_vars = [var for var in t_vars if 'de_' in var.name]
        enc_vars = [var for var in t_vars if 'en_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.dec_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.dec_loss, var_list=dec_vars)
            self.enc_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                .minimize(self.enc_loss, var_list=enc_vars)

        # for test 由噪声生成一张图片
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        # Summary的含义是将参数打包后用于tensorboard的观察和模型的记录
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_dec_sum = tf.summary.scalar("d_loss_fake_dec", d_loss_fake_dec)
        d_loss_fake_dec_ns_sum = tf.summary.scalar("d_loss_fake_dec_ns", d_loss_fake_dec_ns)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        # 创建高斯分布z，均值为0，方差为1
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                    batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                # 创建高斯分布z，均值为0，方差为1
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update D network sess.run喂入数据优化更新D网络
                _,  d_loss = self.sess.run([self.d_optim,  self.d_loss],
                                        feed_dict={self.inputs: batch_images, self.z: batch_z})

                # update decoder network
                _, summary_str, dec_loss, nll_loss, kl_loss = self.sess.run([self.dec_optim, self.merged_summary_op,
                                                        self.dec_loss,self.neg_loglikelihood, self.KL_divergence],
                                                         feed_dict={self.inputs: batch_images, self.z: batch_z})

                # update decoder network
                _, enc_loss = self.sess.run([self.enc_optim, self.enc_loss],
                                                        feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                if np.mod(counter, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, dec_loss: %.8f, enc_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, dec_loss, enc_loss))

                # save training results for every 300 steps 训练300步保存一张生成图片
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = prior.gaussian(self.batch_size, self.z_dim)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch +
                    '_test_all_classes.png')

    @property
    # 加载创建固定模型下的路径，本处为VAE下的训练
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    # 本函数的目的是在于保存训练模型后的checkpoint
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    # 本函数的意义在于读取训练好的模型参数的checkpoint
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train_check(self):
        import re
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            start_epoch = (int)(counter / self.num_batches)
        if start_epoch == self.epoch:
            print(" [*] Training already finished! Begin to test your model")
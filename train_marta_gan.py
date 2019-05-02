from net.marta_gan import MartaGan

if __name__ == '__main__':
  marta_gan = MartaGan(dataset_path="./dataset/uc_train_256", batch_size=64, learning_rate=0.0002)
  marta_gan.train(epoch=10000, load_epoch=0)

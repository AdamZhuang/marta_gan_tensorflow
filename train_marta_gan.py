from net.marta_gan import MartaGan

if __name__ == '__main__':
  marta_gan = MartaGan(dataset_path="./dataset/test", batch_size=1, learning_rate=0.0001)
  marta_gan.train(epoch=10000, load_epoch=100)

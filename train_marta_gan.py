from net.marta_gan import MartaGan

if __name__ == '__main__':
  marta_gan = MartaGan(dataset_path="./dataset/airplane", batch_size=10)
  marta_gan.train(epoch=200, load_epoch=60)

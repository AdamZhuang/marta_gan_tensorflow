from net.marta_gan import MartaGan

if __name__ == '__main__':
  marta_gan = MartaGan(dataset_path="./dataset/uc_train_256_data", batch_size=128, learning_rate=0.0002)
  marta_gan.train(epoch=10000, load_epoch=300)
  # marta_gan.generate_image(2000)

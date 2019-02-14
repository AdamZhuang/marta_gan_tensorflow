from net.marta_gan import MartaGan

if __name__ == '__main__':
  marta_gan = MartaGan(dataset_path="./dataset/uc_train_256_data", batch_size=64, learning_rate=0.0001)
  marta_gan.train(epoch=10000, load_epoch=3050)
  # marta_gan.generate_image(2000)

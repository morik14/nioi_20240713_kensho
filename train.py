import numpy as np
import pandas as pd
import mlflow
import hydra
from omegaconf import DictConfig
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout


def load_sense_dataset(
    data_folder,
    sense_score_filename,
    sense_score_cols,
    image_width,
    image_height,
    channels,
):
    df_sense = pd.read_csv(f"{data_folder}/{sense_score_filename}")
    image_arrays = np.ndarray(
        shape=(len(df_sense), image_width, image_height, channels),
        dtype=np.float32,
    )
    sense_scores = df_sense[sense_score_cols].values
    for i, rows in df_sense.iterrows():
        category = rows["category"]
        filename = rows["filename"]
        img = load_img(
            f"{data_folder}/{category}/{filename}",
            color_mode="rgb",
            target_size=(image_width, image_height),
        )
        image_arrays[i] = img_to_array(img)

    return image_arrays, sense_scores


def compare_TV(history):
    import matplotlib.pyplot as plt

    # Setting Parameters
    mae = history.history["mae"]
    val_mae = history.history["val_mae"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(mae))

    # 1) maeracy Plt
    plt.plot(epochs, mae, "bo", label="training mae")
    plt.plot(epochs, val_mae, "b", label="validation mae")
    plt.title("Training and Validation mae")
    plt.legend()

    plt.figure()

    # 2) Loss Plt
    plt.plot(epochs, loss, "bo", label="training loss")
    plt.plot(epochs, val_loss, "b", label="validation loss")
    plt.title("Training and Validation loss")
    plt.legend()

    # plt.show()


@hydra.main(config_name="config", version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "train_folder": cfg.train_folder,
                "test_folder": cfg.test_folder,
            }
        )
        mlflow.log_params(
            {
                "image_width": cfg.image_width,
                "image_height": cfg.image_height,
                "channels": cfg.channels,
            }
        )
        X_train, y_train = load_sense_dataset(
            cfg.train_folder,
            "sense_score.csv",
            cfg.sense_score_cols,
            cfg.image_width,
            cfg.image_height,
            cfg.channels,
        )
        X_test, y_test = load_sense_dataset(
            cfg.test_folder,
            "sense_score.csv",
            cfg.sense_score_cols,
            cfg.image_width,
            cfg.image_height,
            cfg.channels,
        )

        # VGG16ベースモデルの構築
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(cfg.image_width, cfg.image_height, cfg.channels),
        )

        # 全結合層の追加
        model = Sequential(
            [
                base_model,
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(2, activation="linear"),  # 出力層：酸味と苦みの数値を予測
            ]
        )

        # ベースモデルの層を固定（転移学習）
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

        # モデルの訓練
        history = model.fit(
            X_train,
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_split=0.2,
        )

        compare_TV(history)

        # モデルの保存
        # model.save("coffee_taste_predictor_vgg16.h5")

        # モデルの評価
        loss, mae = model.evaluate(X_test, y_test)
        mlflow.log_metric("mae", mae)

        # y_pred = model.predict(X_test)


if __name__ == "__main__":
    main()

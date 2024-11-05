# for i, path_image in enumerate(path_images):
#     with tifffile.TiffFile(path_image) as tif:
#         for page in tif.pages:
#             for tag in page.tags:
#                 #print(tag.name)
#                 print(tag.name, tag.value)
    #input_filename = inputs_paths[i].stem
    #input_path = str(inputs_paths[i])
    #output_filename = output_paths[i].stem

    # with tifffile.TiffFile(input_path) as tif:
    #     metadata = tif.pages[0].tags
    #     for tag in metadata.values():
    #         print(tag.name, tag.value)
    #
    #
    # input_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    # out_image = cv2.imread(str(output_paths[i]), cv2.IMREAD_UNCHANGED)
    #
    # b, r, g, a = cv2.split(input_image)
    # input_image = cv2.merge((r, g, b)) / 255
    # out_image = out_image/255
    # fig = plt.figure(figsize=(7, 9))
    # axs = [fig.add_subplot(1, 2, 1),
    #        fig.add_subplot(1, 2, 2)]
    # axs[0].imshow(input_image)
    # axs[1].imshow(out_image)
    # axs[0].set_title(input_filename)
    # axs[1].set_title(output_filename)
    # plt.show()
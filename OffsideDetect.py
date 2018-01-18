import cv2
import numpy as np

mouse_points = []
pitch_points = []
image_points = []


def get_matchings(image, pitch):
    print("\n\nIn this step, you are given two images and asked to click on the points"
          "that are visible and mutual in two images. You can click either one by one or"
          "group by group, but the order is important. \n!!! Press ESC when you are done !!!\nIf "
          "pairs are not equal in length you are going to be prompted again.")
    input("Press enter when ready")

    while 1:
        cv2.namedWindow('Pitch Coordinates')
        cv2.setMouseCallback('Pitch Coordinates', get_pitch_coordinates, param=pitch)
        cv2.imshow('Pitch Coordinates', pitch)

        cv2.namedWindow('Image Coordinates')
        cv2.setMouseCallback('Image Coordinates', get_image_coordinates, param=image)
        cv2.imshow('Image Coordinates', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return image_points, pitch_points


def get_user_points(image, num_of_points):
    print("\n\nOn the following image, click on the attacker's and the defender's positions"
          "in same order specified.")
    input("Press enter when ready")

    while 1 and len(mouse_points) < num_of_points:
        global pitch_points, image_points
        pitch_points = []
        image_points = []
        cv2.namedWindow('Player Coordinates')
        cv2.setMouseCallback('Player Coordinates', get_mouse_coordinates, param=image)
        cv2.imshow('Player Coordinates', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    print("Points selected!")
    cv2.destroyAllWindows()
    return mouse_points


def determine_offside(x1, x2):
    offside = False
    # Attack is to the right
    if x1 > 480:
        offside = True if x1 > x2 else False
    else:
        offside = True if x2 > x1 else False
    return offside


def draw_on_pitch(pitch, output_points):
    line_points = []
    normals = []
    for point in output_points:
        x, y = point
        x = int(x)
        y = int(y)
        line_points.append([x, 17])
        line_points.append([x, 584])
        normals.append([x, y, 0])
        normals.append([x, y, 10])
        cv2.rectangle(pitch, (x, y), (x + 4, y - 4), [0, 0, 255])
    return line_points, normals


def draw_from_pitch_to_image(image, reverse_output_points):
    for i in range(0, len(reverse_output_points), 2):
        x1, y1 = reverse_output_points[i]
        x2, y2 = reverse_output_points[i + 1]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        color = [255, 0, 0] if i < 1 else [0, 0, 255]
        cv2.line(image, (x1, y1), (x2, y2), color, 2)


def main():
    print('This program reads a image file and takes multiple point pairs from '
          'two football field images. Then asks for the attacker and defender\'s position.'
          'Finally it draws perpendicular and parallel lines to field ground with '
          'the distance and offside decision on bottom left.')

    file = input('Type file path including file name: ')
    filename = file.split('/')[-1]
    pitch_file = './pitch.png'
    image = cv2.imread(file)
    pitch = cv2.imread(pitch_file)

    # Get user's points
    user_points = np.array(get_user_points(image, 2), dtype='float32')
    cv2.destroyAllWindows()

    pitch_height = 567
    pitch_width = 876

    image_matching = None
    pitch_matching = None

    isMathingsOk = False
    while not isMathingsOk:
        copy_image = np.copy(image)
        copy_pitch = np.copy(pitch)
        image_matching, pitch_matching = get_matchings(copy_image, copy_pitch)
        if len(image_matching) > 3 and len(image_matching) == len(pitch_matching):
            print("Got {0} pairs.".format(len(image_matching)))
            break
        global pitch_points, image_points
        pitch_points = []
        image_points = []

        print("Need at least four pairs")

    print("\n\n{0} pairs selected.".format(len(image_matching)))

    # Get pitch dimensions
    length = None
    height = None

    while length is None or height is None:
        try:
            length_string = input("Please write the field's width (default: 90) [90m, 120m]: ")
            length_string = "90" if not length_string else length_string
            length = int(length_string)

            height_string = input("Please write the field's height (default: 45) [45m, 90m]: ")
            height_string = "45" if not height_string else height_string
            height = int(height_string)

            break
        except ValueError:
            print('Wrong value type :(, try again.')
            pass

    # Get distance per pixel values
    width_per_pixel = length / pitch_width
    height_per_pixel = height / pitch_height

    # Convert python arrays to numpy arrays
    image_matching = np.array(image_matching)
    pitch_matching = np.array(pitch_matching)

    # Find matching between given points
    h, status = cv2.findHomography(image_matching, pitch_matching)

    # Map user selected players' coordinates to pitch image
    input_coor = np.array([user_points])
    output = cv2.perspectiveTransform(input_coor, h)[0]
    line_points, normals = draw_on_pitch(pitch, output)

    length = np.absolute(line_points[0][0] - line_points[2][0]) * width_per_pixel
    offside = determine_offside(line_points[0][0], line_points[2][0])

    # Get line points from pitch image
    reverse_output = cv2.perspectiveTransform(np.array([line_points], dtype='float32'), np.linalg.inv(h))[0]
    draw_from_pitch_to_image(image, reverse_output)

    # Show pitch with point locations
    cv2.imshow('Result', pitch)

    # Show image with lines, offside decision and distance
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_h, image_w, ch = image.shape

    cv2.putText(image, 'Distance: {0:.2f} metres'.format(length),
                (10, image_h-32), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    offside_color = [255, 0, 0] if not offside else [0, 0, 255]
    cv2.putText(image, 'Offside: {0}'.format(offside),
                (10, image_h-15), font, 0.5, offside_color, 1, cv2.LINE_AA)
    cv2.imshow('Result2', image)
    cv2.waitKey(0)
    cv2.imwrite('./outputs/' + filename, image)
    cv2.imwrite('./outputs/pitch_'+filename, pitch)
    print("Offside: {0}\tDistance between lines: {1}".format(offside, length))


def get_mouse_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_points.append([x, y])


def get_pitch_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(params, (x-2, y-2), (x + 2, y + 2), [0, 0, 255])
        pitch_points.append([x, y])


def get_image_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(params, (x-2, y-2), (x + 2, y + 2), [0, 0, 255])
        image_points.append([x, y])


if __name__ == '__main__':
    main()

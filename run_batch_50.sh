#!/bin/bash

echo "🚀 Running 50 image editing examples..."

python -m atomic_edits.pipeline.cli_e2e -i images/adventure.jpg -t "make the sky deep blue and increase the brightness of the rocks" -o artifacts/examples/adventure_edit
python -m atomic_edits.pipeline.cli_e2e -i images/animals.jpg -t "make the cat orange and make the dog black and white" -o artifacts/examples/animals_edit
python -m atomic_edits.pipeline.cli_e2e -i images/billboard.jpg -t "replace the billboard text with OPEN and increase the brightness" -o artifacts/examples/billboard_edit
python -m atomic_edits.pipeline.cli_e2e -i images/bottle.jpg -t "make the bottle green and blur the background" -o artifacts/examples/bottle_edit
python -m atomic_edits.pipeline.cli_e2e -i images/car.jpg -t "change the car color to blue and darken the trees" -o artifacts/examples/car_edit
python -m atomic_edits.pipeline.cli_e2e -i images/city.jpg -t "make the bicycle red and increase the sharpness" -o artifacts/examples/city_edit
python -m atomic_edits.pipeline.cli_e2e -i images/coffeemug.jpg -t "turn the mug green and blur the background" -o artifacts/examples/coffeemug_edit
python -m atomic_edits.pipeline.cli_e2e -i images/corner.jpg -t "make the chair black and increase the brightness of the plant" -o artifacts/examples/corner_edit
python -m atomic_edits.pipeline.cli_e2e -i images/flower.jpg -t "make the petals pink and darken the background" -o artifacts/examples/flower_edit
python -m atomic_edits.pipeline.cli_e2e -i images/hair.jpg -t "make the hair blonde and blur the background" -o artifacts/examples/hair_edit
python -m atomic_edits.pipeline.cli_e2e -i images/laptop.jpg -t "add a blue sticker to the laptop and increase the brightness" -o artifacts/examples/laptop_edit
python -m atomic_edits.pipeline.cli_e2e -i images/mountains.jpg -t "make the sky sunset colors and increase the saturation of the grass" -o artifacts/examples/mountains_edit
python -m atomic_edits.pipeline.cli_e2e -i images/mountains2.jpg -t "make the water more vibrant blue and increase the brightness" -o artifacts/examples/mountains2_edit
python -m atomic_edits.pipeline.cli_e2e -i images/orchids.jpg -t "make the orchids white and darken the background" -o artifacts/examples/orchids_edit
python -m atomic_edits.pipeline.cli_e2e -i images/pumpkin.jpg -t "make the pumpkin more orange and blur the background" -o artifacts/examples/pumpkin_edit
python -m atomic_edits.pipeline.cli_e2e -i images/ship.jpg -t "make the ship blue and increase the brightness of the sky" -o artifacts/examples/ship_edit
python -m atomic_edits.pipeline.cli_e2e -i images/shirt.png -t "make the shirt blue and remove the logo" -o artifacts/examples/shirt_edit
python -m atomic_edits.pipeline.cli_e2e -i images/sunflower.jpg -t "increase the saturation of the sunflowers and blur the background" -o artifacts/examples/sunflower_edit
python -m atomic_edits.pipeline.cli_e2e -i images/trees.jpg -t "make the sky sunset colors and increase the brightness of the sand" -o artifacts/examples/trees_edit
python -m atomic_edits.pipeline.cli_e2e -i images/underwater.jpg -t "make the flowers more vibrant and darken the background" -o artifacts/examples/underwater_edit
python -m atomic_edits.pipeline.cli_e2e -i images/bananas.jpg -t "make the background blue and make the bananas more yellow" -o artifacts/examples/bananas_edit

python -m atomic_edits.pipeline.cli_e2e -i images/bird.jpg -t "make the bird red and blue and darken the background" -o artifacts/examples/bird_edit

python -m atomic_edits.pipeline.cli_e2e -i images/bus.jpg -t "make the bus red and brighten the sky" -o artifacts/examples/bus_edit

python -m atomic_edits.pipeline.cli_e2e -i images/cake.jpg -t "make the cake pink and add more blueberries on top" -o artifacts/examples/cake_edit

python -m atomic_edits.pipeline.cli_e2e -i images/chairs.jpg -t "make the chair blue and darken the floor" -o artifacts/examples/chairs_edit

python -m atomic_edits.pipeline.cli_e2e -i images/clock.jpg -t "make the clock black and brighten the wall" -o artifacts/examples/clock_edit

python -m atomic_edits.pipeline.cli_e2e -i images/cola.jpg -t "make the bottle green and brighten the background" -o artifacts/examples/cola_edit

python -m atomic_edits.pipeline.cli_e2e -i images/dog.jpg -t "make the dog brown and brighten the background" -o artifacts/examples/dog_edit

python -m atomic_edits.pipeline.cli_e2e -i images/dress.jpg -t "make the dress blue and darken the background" -o artifacts/examples/dress_edit

python -m atomic_edits.pipeline.cli_e2e -i images/green-car.jpg -t "make the car red and brighten the road" -o artifacts/examples/green_car_edit

python -m atomic_edits.pipeline.cli_e2e -i images/lavender.jpg -t "make the flowers pink and brighten the sky" -o artifacts/examples/lavender_edit

python -m atomic_edits.pipeline.cli_e2e -i images/papaya.jpg -t "make the papaya more orange and darken the leaf" -o artifacts/examples/papaya_edit

python -m atomic_edits.pipeline.cli_e2e -i images/person-hat.jpg -t "make the shirt blue and make the hat white" -o artifacts/examples/person_hat_edit

python -m atomic_edits.pipeline.cli_e2e -i images/rabbit.jpg -t "make the rabbit gray and blur the background" -o artifacts/examples/rabbit_edit

python -m atomic_edits.pipeline.cli_e2e -i images/red-car.jpg -t "make the car blue and brighten the background" -o artifacts/examples/red_car_edit

python -m atomic_edits.pipeline.cli_e2e -i images/rose.jpg -t "make the rose pink and darken the background" -o artifacts/examples/rose_edit

python -m atomic_edits.pipeline.cli_e2e -i images/shoes.jpg -t "make the shoes red and brighten the stairs" -o artifacts/examples/shoes_edit

python -m atomic_edits.pipeline.cli_e2e -i images/tulips.jpg -t "make the tulips yellow and brighten the background" -o artifacts/examples/tulips_edit

python -m atomic_edits.pipeline.cli_e2e -i images/vase.jpg -t "make the vase blue and make the oranges more vibrant" -o artifacts/examples/vase_edit

python -m atomic_edits.pipeline.cli_e2e -i images/yellow-van.jpg -t "make the van blue and darken the sky" -o artifacts/examples/yellow_van_edit

python -m atomic_edits.pipeline.cli_e2e -i images/cctv.jpg -t "make the wall blue and increase the brightness of the flowers" -o artifacts/examples/cctv_edit

python -m atomic_edits.pipeline.cli_e2e -i images/cup.jpg -t "make the cup green and increase the saturation of the leaves" -o artifacts/examples/cup_edit

python -m atomic_edits.pipeline.cli_e2e -i images/dog-ball.jpg -t "make the ball yellow and darken the background" -o artifacts/examples/dog_ball_edit

python -m atomic_edits.pipeline.cli_e2e -i images/empire-state.jpg -t "make the sky sunset colors and increase the contrast" -o artifacts/examples/empire_state_edit

python -m atomic_edits.pipeline.cli_e2e -i images/jeep.jpg -t "make the field more vibrant green and increase the brightness" -o artifacts/examples/jeep_edit

python -m atomic_edits.pipeline.cli_e2e -i images/laptop2.jpg -t "make the blanket blue and increase the brightness of the screen" -o artifacts/examples/laptop2_edit

python -m atomic_edits.pipeline.cli_e2e -i images/oreo.jpg -t "increase the contrast of the cookies and blur the background slightly" -o artifacts/examples/oreo_edit

python -m atomic_edits.pipeline.cli_e2e -i images/person.jpg -t "make the shirt blue and increase the saturation of the grass" -o artifacts/examples/person_edit

python -m atomic_edits.pipeline.cli_e2e -i images/skyline.jpg -t "make the sky more vibrant orange and increase the overall contrast" -o artifacts/examples/skyline_edit

python -m atomic_edits.pipeline.cli_e2e -i images/sunflower-person.jpg -t "make the sunflower more vibrant yellow and blur the background" -o artifacts/examples/sunflower_person_edit

echo "✅ All 50 examples completed!"
echo "📁 Results saved in artifacts/examples/"
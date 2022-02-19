import examples.tested.lib as tl


def main():
    civic = tl.Car("civic", 10000)
    civic.save()

    miata = tl.Car("miata", 20000)
    miata.save()

    value = tl.add_slow(civic.price, miata.price)
    return value


if __name__ == "__main__":
    main()
